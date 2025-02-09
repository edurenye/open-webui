import asyncio
import base64
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS, SRC_LOG_LEVELS
from open_webui.routers.files import upload_file
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.images.comfyui import (
    ComfyUIGenerateImageForm,
    ComfyUIWorkflow,
    comfyui_generate_image,
    ComfyUINodeInput,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["IMAGES"])

IMAGE_CACHE_DIR = CACHE_DIR / "image" / "generations"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter()


@router.get("/config")
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
        "default_engine": request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE,
        "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
        "openai": request.app.state.config.OPENAI_IMAGE_CONFIG,
        "automatic1111": request.app.state.config.AUTOMATIC1111_CONFIG,
        "comfyui": request.app.state.config.COMFYUI_CONFIG,
        "gemini": request.app.state.config.GEMINI_CONFIG,
    }

class AccessControlDict(BaseModel):
    group_ids: list[str] = []

class AccessControl(BaseModel):
    read: AccessControlDict = AccessControlDict()
    write: AccessControlDict = AccessControlDict()

class WorkflowNode(BaseModel):
    type: str
    key: str
    node_ids: str

class OpenAIModelWrapperConfigForm(BaseModel):
    id: str
    name: str
    model: str
    is_default: bool
    enabled: bool
    image_size: str
    image_steps: int
    access_control: Optional[AccessControl]

class OpenAIConfigForm(BaseModel):
    model_config = {"protected_namespaces": ()}
    api_base_url: str
    api_key: str
    model_wrappers: list[OpenAIModelWrapperConfigForm]

class Automatic1111ModelWrapperConfigForm(BaseModel):
    id: str
    name: str
    model: str
    is_default: bool
    enabled: bool
    image_size: str
    image_steps: int
    access_control: Optional[AccessControl]
    cfg_scale: Optional[float]
    sampler: Optional[str]
    scheduler: Optional[str]

class Automatic1111ConfigForm(BaseModel):
    model_config = {"protected_namespaces": ()}
    api_base_url: str
    api_key: str
    model_wrappers: list[Automatic1111ModelWrapperConfigForm]

class ComfyUIModelWrapperConfigForm(BaseModel):
    id: str
    name: str
    model: str
    is_default: bool
    enabled: bool
    image_size: str
    image_steps: int
    access_control: Optional[AccessControl]
    workflow: Optional[str]
    workflow_nodes: Optional[list[WorkflowNode]]

class ComfyUIConfigForm(BaseModel):
    model_config = {"protected_namespaces": ()}
    api_base_url: str
    api_key: str
    model_wrappers: list[ComfyUIModelWrapperConfigForm]

class GeminiModelWrapperConfigForm(BaseModel):
    id: str
    name: str
    model: str
    is_default: bool
    enabled: bool
    image_size: str
    image_steps: int
    access_control: Optional[AccessControl]

class GeminiConfigForm(BaseModel):
    model_config = {"protected_namespaces": ()}
    api_base_url: str
    api_key: str
    model_wrappers: list[GeminiModelWrapperConfigForm]

class ConfigForm(BaseModel):
    enabled: bool
    default_engine: str
    prompt_generation: bool
    openai: OpenAIConfigForm
    automatic1111: Automatic1111ConfigForm
    comfyui: ComfyUIConfigForm
    gemini: GeminiConfigForm


@router.post("/config/update")
async def update_config(
    request: Request, form_data: ConfigForm, user=Depends(get_admin_user)
):
    try:
        request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE = form_data.default_engine
        request.app.state.config.ENABLE_IMAGE_GENERATION = form_data.enabled
        request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = form_data.prompt_generation

        openai_config = form_data.openai.model_dump()
        automatic1111_config = form_data.automatic1111.model_dump()
        comfyui_config = form_data.comfyui.model_dump()
        gemini_config = form_data.gemini.model_dump()

        request.app.state.config.OPENAI_IMAGE_CONFIG = openai_config
        request.app.state.config.AUTOMATIC1111_CONFIG = automatic1111_config
        request.app.state.config.COMFYUI_CONFIG = comfyui_config
        request.app.state.config.GEMINI_CONFIG = gemini_config

        return {
            "enabled": request.app.state.config.ENABLE_IMAGE_GENERATION,
            "default_engine": request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE,
            "prompt_generation": request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION,
            "openai": openai_config,
            "automatic1111": automatic1111_config,
            "comfyui": comfyui_config,
            "gemini": gemini_config,
        }
    except Exception as e:
        log.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_automatic1111_api_key(request: Request):
    if request.app.state.config.AUTOMATIC1111_CONFIG["api_key"] is None:
        return ""
    else:
        auth1111_byte_string = request.app.state.config.AUTOMATIC1111_CONFIG["api_key"].encode(
            "utf-8"
        )
        auth1111_base64_encoded_bytes = base64.b64encode(auth1111_byte_string)
        auth1111_base64_encoded_string = auth1111_base64_encoded_bytes.decode("utf-8")
        return f"Basic {auth1111_base64_encoded_string}"


@router.get("/config/{engine}/url/verify")
async def verify_url(engine: str, request: Request, user=Depends(get_admin_user)):
    try:
        if engine == "automatic1111":
            try:
                r = requests.get(
                    url=f"{request.app.state.config.AUTOMATIC1111_CONFIG['api_base_url']}/sdapi/v1/options",
                    headers={"authorization": get_automatic1111_api_key(request)},
                )
                r.raise_for_status()
                return True
            except Exception as e:
                log.error(f"Error verifying Automatic1111 URL: {str(e)}")
                raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)

        elif engine == "comfyui":
            try:
                headers = {}
                if request.app.state.config.COMFYUI_CONFIG.get('api_key'):
                    headers["Authorization"] = f"Bearer {request.app.state.config.COMFYUI_CONFIG['api_key']}"

                r = requests.get(
                    url=f"{request.app.state.config.COMFYUI_CONFIG['api_base_url']}/object_info",
                    headers=headers,
                )
                r.raise_for_status()
                return True
            except Exception as e:
                log.error(f"Error verifying ComfyUI URL: {str(e)}")
                raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)

        elif engine == "openai":
            try:
                headers = {
                    "Authorization": f"Bearer {request.app.state.config.OPENAI_IMAGE_CONFIG['api_key']}",
                    "Content-Type": "application/json"
                }

                r = requests.get(
                    url=f"{request.app.state.config.OPENAI_IMAGE_CONFIG['api_base_url']}/models",
                    headers=headers,
                )
                r.raise_for_status()
                return True
            except Exception as e:
                log.error(f"Error verifying OpenAI URL: {str(e)}")
                raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
        elif engine == "gemini":
            try:
                headers = {
                    "Authorization": f"Bearer {request.app.state.config.GEMINI_IMAGE_CONFIG['api_key']}",
                    "Content-Type": "application/json"
                }

                r = requests.get(
                    url=f"{request.app.state.config.GEMINI_IMAGE_CONFIG['api_base_url']}/models",
                    headers=headers,
                )
                r.raise_for_status()
                return True
            except Exception as e:
                log.error(f"Error verifying Gemini URL: {str(e)}")
                raise HTTPException(status_code=400, detail=ERROR_MESSAGES.INVALID_URL)
        else:
            raise HTTPException(status_code=400, detail="Invalid engine specified")

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in verify_url: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_default_model(request, engine=None):
    """Get the default model for a given engine or the current engine"""
    engine = engine or request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE

    if engine == "openai":
        model_wrappers = request.app.state.config.OPENAI_IMAGE_CONFIG.get("model_wrappers")
    elif engine == "automatic1111":
        model_wrappers = request.app.state.config.AUTOMATIC1111_CONFIG.get("model_wrappers")
    elif engine == "comfyui":
        model_wrappers = request.app.state.config.COMFYUI_CONFIG.get("model_wrappers")
    elif engine == "gemini":
        model_wrappers = request.app.state.config.GEMINI_CONFIG.get("model_wrappers")
    else:
        return None

    for model_wrapper in model_wrappers:
        if model_wrapper["is_default"]:
            return model_wrapper["model"]

    return model_wrappers[0].model if model_wrappers else None


def set_image_model(request: Request, model: str):
    log.info(f"Setting image model to {model}")
    if request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE in ["", "automatic1111"]:
        api_key = get_automatic1111_api_key(request)
        r = requests.get(
            url=f"{request.app.state.config.AUTOMATIC1111_CONFIG['api_base_url']}/sdapi/v1/options",
            headers={"authorization": api_key},
        )
        options = r.json()
        if model != options["sd_model_checkpoint"]:
            options["sd_model_checkpoint"] = model
            r = requests.post(
                url=f"{request.app.state.config.AUTOMATIC1111_CONFIG['api_base_url']}/sdapi/v1/options",
                json=options,
                headers={"authorization": api_key},
            )
    return model


def get_image_model(request):
    default_engine = request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE

    if default_engine == "openai":
        return get_default_model(request) or "dall-e-2"
    elif default_engine == "gemini":
        return get_default_model(request) or "imagen-3.0-generate-002"
    elif default_engine == "comfyui":
        return get_default_model(request) or ""
    elif default_engine in ["automatic1111", ""]:
        try:
            r = requests.get(
                url=f"{request.app.state.config.AUTOMATIC1111_CONFIG['api_base_url']}/sdapi/v1/options",
                headers={"authorization": get_automatic1111_api_key(request)},
            )
            options = r.json()
            return options["sd_model_checkpoint"]
        except Exception as e:
            request.app.state.config.ENABLE_IMAGE_GENERATION = False
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class ImageConfigForm(BaseModel):
    MODEL: str
    IMAGE_SIZE: str
    IMAGE_STEPS: int


@router.get("/image/config")
async def get_image_config(request: Request, user=Depends(get_admin_user)):
    """Get the current image generation configuration"""
    log.error("get_image_config")
    log.error(json.dumps(request.app.state.config.__dict__, indent=2, default=str))
    model_id = get_default_model(request)
    default_engine = request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE

    if default_engine == "openai":
        model_wrappers = request.app.state.config.OPENAI_IMAGE_CONFIG["model_wrappers"]
    if default_engine == "gemini":
        model_wrappers = request.app.state.config.GEMINI_IMAGE_CONFIG["model_wrappers"]
    elif default_engine == "automatic1111":
        model_wrappers = request.app.state.config.AUTOMATIC1111_CONFIG["model_wrappers"]
    elif default_engine == "comfyui":
        model_wrappers = request.app.state.config.COMFYUI_CONFIG["model_wrappers"]
    else:
        model_wrappers = []

    model_config = next((m for m in model_wrappers if m["model"] == model_id), model_wrappers[0] if model_wrappers else None)

    return {
        "MODEL": model_id,
        "IMAGE_SIZE": model_config["image_size"] if model_config else "512x512",
        "IMAGE_STEPS": model_config["image_steps"] if model_config else 50,
    }


@router.post("/image/config/update")
async def update_image_config(
    request: Request, form_data: ImageConfigForm, user=Depends(get_admin_user)
):
    default_engine = request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE

    pattern = r"^\d+x\d+$"
    if not re.match(pattern, form_data.IMAGE_SIZE):
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 512x512)."),
        )

    if form_data.IMAGE_STEPS < 0:
        raise HTTPException(
            status_code=400,
            detail=ERROR_MESSAGES.INCORRECT_FORMAT("  (e.g., 50)."),
        )

    if default_engine == "openai":
        model_wrappers = request.app.state.config.OPENAI_IMAGE_CONFIG["model_wrappers"]
    elif default_engine == "gemini":
        model_wrappers = request.app.state.config.GEMINI_IMAGE_CONFIG["model_wrappers"]
    elif default_engine == "automatic1111":
        model_wrappers = request.app.state.config.AUTOMATIC1111_CONFIG["model_wrappers"]
    elif default_engine == "comfyui":
        model_wrappers = request.app.state.config.COMFYUI_CONFIG["model_wrappers"]
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid engine"
        )

    model = next((m for m in model_wrappers if m["model"] == form_data.MODEL), None)
    if not model:
        raise HTTPException(
            status_code=400,
            detail=f"Model {form_data.MODEL} not found"
        )

    model["image_size"] = form_data.IMAGE_SIZE
    model["image_steps"] = form_data.IMAGE_STEPS

    if default_engine in ["automatic1111", ""]:
        set_image_model(request, form_data.MODEL)

    return {
        "MODEL": form_data.MODEL,
        "IMAGE_SIZE": form_data.IMAGE_SIZE,
        "IMAGE_STEPS": form_data.IMAGE_STEPS,
    }


@router.get("/engines/{engine}/models")
def get_engine_models(engine: str, request: Request, user=Depends(get_verified_user)):
    try:
        log.error(f"Getting models for engine: {engine}")
        if engine == "openai":
            return [
                {"id": "dall-e-2", "name": "DALL·E 2"},
                {"id": "dall-e-3", "name": "DALL·E 3"},
            ]
        elif engine == "gemini":
            return [
                {"id": "imagen-3-0-generate-002", "name": "imagen-3.0 generate-002"},
            ]
        elif engine == "comfyui":
            base_url = request.app.state.config.COMFYUI_CONFIG.get('api_base_url')
            api_key = request.app.state.config.COMFYUI_CONFIG.get('api_key')

            if not base_url:
                raise HTTPException(
                    status_code=400,
                    detail="ComfyUI base URL not configured"
                )

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            try:
                r = requests.get(
                    url=f"{base_url}/object_info",
                    headers=headers,
                )
                r.raise_for_status()
                info = r.json()

                return list(
                    map(
                        lambda model: {"id": model, "name": model},
                        info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0],
                    )
                )
            except Exception as e:
                log.error(f"ComfyUI request failed: {str(e)}")
                raise
        elif engine == "automatic1111":
            base_url = request.app.state.config.AUTOMATIC1111_CONFIG.get('api_base_url')
            if not base_url:
                raise HTTPException(
                    status_code=400,
                    detail="Automatic1111 base URL not configured"
                )

            r = requests.get(
                url=f"{base_url}/sdapi/v1/sd-models",
                headers={"authorization": get_automatic1111_api_key(request)},
            )
            models = r.json()
            return list(
                map(
                    lambda model: {"id": model["title"], "name": model["model_name"]},
                    models,
                )
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid engine")
    except Exception as e:
        log.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(e))


class GenerateImageForm(BaseModel):
    model_config = {"protected_namespaces": ()}
    model: Optional[str] = None
    prompt: str
    size: Optional[str] = None
    steps: Optional[int] = None
    n: int = 1
    negative_prompt: Optional[str] = None
    engine: Optional[str] = None
    model_wrapper_id: Optional[str] = None


def load_b64_image_data(b64_str):
    try:
        if "," in b64_str:
            header, encoded = b64_str.split(",", 1)
            mime_type = header.split(";")[0]
            img_data = base64.b64decode(encoded)
        else:
            mime_type = "image/png"
            img_data = base64.b64decode(b64_str)
        return img_data, mime_type
    except Exception as e:
        log.exception(f"Error loading image data: {e}")
        return None


def load_url_image_data(url, headers=None):
    try:
        if headers:
            r = requests.get(url, headers=headers)
        else:
            r = requests.get(url)

        r.raise_for_status()
        if r.headers["content-type"].split("/")[0] == "image":
            mime_type = r.headers["content-type"]
            return r.content, mime_type
        else:
            log.error("Url does not point to an image.")
            return None

    except Exception as e:
        log.exception(f"Error saving image: {e}")
        return None


def upload_image(request, image_metadata, image_data, content_type, user):
    image_format = mimetypes.guess_extension(content_type)
    file = UploadFile(
        file=io.BytesIO(image_data),
        filename=f"generated-image{image_format}",  # will be converted to a unique ID on upload_file
        headers={
            "content-type": content_type,
        },
    )
    file_item = upload_file(request, file, user, file_metadata=image_metadata)
    url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
    return url


def has_read_access(user, model_wrapper):
    if user.role == "admin":
        return True

    if not model_wrapper.get("access_control"):
        return True

    user_groups = set(user.groups)
    allowed_groups = set(model_wrapper["access_control"]["read"]["group_ids"])
    return bool(user_groups & allowed_groups)


@router.post("/generations")
async def image_generations(
    request: Request,
    form_data: GenerateImageForm,
    user=Depends(get_verified_user),
):
    try:
        engine = form_data.engine or request.app.state.config.DEFAULT_IMAGE_GENERATION_ENGINE

        if engine == "openai":
            model_wrappers = request.app.state.config.OPENAI_IMAGE_CONFIG["model_wrappers"]
        elif engine == "gemini":
            model_wrappers = request.app.state.config.GEMINI_IMAGE_CONFIG["model_wrappers"]
        elif engine == "automatic1111":
            model_wrappers = request.app.state.config.AUTOMATIC1111_CONFIG["model_wrappers"]
        elif engine == "comfyui":
            model_wrappers = request.app.state.config.COMFYUI_CONFIG["model_wrappers"]
        else:
            log.error(f"Invalid engine: {engine}")
            raise HTTPException(status_code=400, detail=f"Invalid engine: {engine}")

        model_wrapper = None
        if form_data.model_wrapper_id:
            model_wrapper = next(
                (m for m in model_wrappers if m["id"] == form_data.model_wrapper_id),
                None
            )
        if not model_wrapper:
            model_wrapper = next(
                (m for m in model_wrappers if m["is_default"]),
                model_wrappers[0] if model_wrappers else None
            )

        if not model_wrapper:
            raise HTTPException(
                status_code=400,
                detail=f"No valid model wrapper found for engine {engine}"
            )

        if not has_read_access(user, model_wrapper):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this model wrapper"
            )

        model = form_data.model or model_wrapper["model"]
        image_size = form_data.size or model_wrapper["image_size"]
        steps = form_data.steps or model_wrapper["image_steps"]

        try:
            width, height = tuple(map(int, image_size.split("x")))
        except (ValueError, KeyError) as e:
            log.error(f"Error parsing model configuration: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model configuration: {str(e)}"
            )

        r = None
        try:
            if engine == "openai":
                headers = {}
                headers["Authorization"] = (
                    f"Bearer {request.app.state.config.OPENAI_IMAGE_CONFIG['api_key']}"
                )
                headers["Content-Type"] = "application/json"

                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers["X-OpenWebUI-User-Name"] = user.name
                    headers["X-OpenWebUI-User-Id"] = user.id
                    headers["X-OpenWebUI-User-Email"] = user.email
                    headers["X-OpenWebUI-User-Role"] = user.role

                api_base_url = request.app.state.config.OPENAI_IMAGE_CONFIG.get('api_base_url')
                if not api_base_url:
                    log.error("OpenAI API base URL not configured")
                    raise HTTPException(
                        status_code=400,
                        detail="OpenAI API base URL not configured"
                    )

                data = {
                    "model": model,
                    "prompt": form_data.prompt,
                    "n": form_data.n,
                    "size": form_data.size if form_data.size else image_size,
                    "response_format": "b64_json",
                }

                r = await asyncio.to_thread(
                    requests.post,
                    url=f"{api_base_url}/images/generations",
                    json=data,
                    headers=headers,
                )

                r.raise_for_status()
                res = r.json()

                images = []

                for image in res["data"]:
                    if image_url := image.get("url", None):
                        image_data, content_type = load_url_image_data(image_url, headers)
                    else:
                        image_data, content_type = load_b64_image_data(image["b64_json"])

                    url = upload_image(request, data, image_data, content_type, user)
                    images.append({"url": url})
                return images

            elif engine == "gemini":
                headers = {}
                headers["Content-Type"] = "application/json"
                headers["x-goog-api-key"] = request.app.state.config.GEMINI_IMAGE_CONFIG['api_key']

                data = {
                    "instances": {"prompt": form_data.prompt},
                    "parameters": {
                        "sampleCount": form_data.n,
                        "outputOptions": {"mimeType": "image/png"},
                    },
                }

                # Use asyncio.to_thread for the requests.post call
                r = await asyncio.to_thread(
                    requests.post,
                    url=f"{request.app.state.config.GEMINI_IMAGE_CONFIG.get('api_base_url')}/models/{model}:predict",
                    json=data,
                    headers=headers,
                )

                r.raise_for_status()
                res = r.json()

                images = []
                for image in res["predictions"]:
                    image_data, content_type = load_b64_image_data(
                        image["bytesBase64Encoded"]
                    )
                    url = upload_image(request, data, image_data, content_type, user)
                    images.append({"url": url})

                return images

            elif engine == "comfyui":
                try:
                    data = {
                        "prompt": form_data.prompt,
                        "width": width,
                        "height": height,
                        "n": form_data.n,
                        "steps": steps,
                    }

                    if form_data.negative_prompt is not None:
                        data["negative_prompt"] = form_data.negative_prompt

                    workflow = model_wrapper.get("workflow")
                    if not workflow:
                        log.error("No workflow defined in model configuration")
                        raise HTTPException(
                            status_code=400,
                            detail="No workflow defined for this model"
                        )

                    workflow_nodes = model_wrapper.get("workflow_nodes", [])
                    if not workflow_nodes:
                        log.error("No workflow nodes defined in model configuration")
                        raise HTTPException(
                            status_code=400,
                            detail="No workflow nodes defined for this model"
                        )

                    processed_nodes = []
                    for node in workflow_nodes:
                        try:
                            if isinstance(node["node_ids"], str):
                                node_ids = [node["node_ids"]]
                            elif isinstance(node["node_ids"], (int, float)):
                                node_ids = [str(node["node_ids"])]
                            else:
                                node_ids = [str(id) for id in node["node_ids"]]

                            node_input = ComfyUINodeInput(
                                type=node.get("type"),
                                key=node.get("key", "text"),
                                node_ids=node_ids,
                                value=node.get("value")
                            )
                            processed_nodes.append(node_input.model_dump())
                        except Exception as e:
                            log.error(f"Error processing workflow node: {str(e)}")
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid workflow node configuration: {str(e)}"
                            )

                    workflow_data = ComfyUIGenerateImageForm(
                        workflow=ComfyUIWorkflow(
                            workflow=workflow,
                            nodes=processed_nodes
                        ),
                        **data
                    )

                    base_url = request.app.state.config.COMFYUI_CONFIG.get("api_base_url")
                    api_key = request.app.state.config.COMFYUI_CONFIG.get("api_key")

                    if not base_url:
                        raise HTTPException(
                            status_code=400,
                            detail="ComfyUI base URL not configured"
                        )

                    res = await comfyui_generate_image(
                        model,
                        workflow_data,
                        user.id,
                        base_url,
                        api_key
                    )

                    if not res or "data" not in res:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to generate images - no data returned from ComfyUI"
                        )
                    log.debug(f"res: {res}")

                    images = []
                    for image in res["data"]:
                        headers = None
                        if api_key:
                            headers = {"Authorization": f"Bearer {api_key}"}

                        image_data, content_type = load_url_image_data(image["url"], headers)
                        url = upload_image(
                            request,
                            form_data.model_dump(exclude_none=True),
                            image_data,
                            content_type,
                            user,
                        )
                        images.append({"url": url})
                    return images

                except HTTPException:
                    raise
                except Exception as e:
                    log.exception("Error in ComfyUI image generation")
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI generation failed: {str(e)}"
                    )
            elif engine in ["automatic1111", ""]:
                base_url = request.app.state.config.AUTOMATIC1111_CONFIG.get('api_base_url')
                if not base_url:
                    log.error("Automatic1111 base URL not configured")
                    raise HTTPException(
                        status_code=400,
                        detail="Automatic1111 base URL not configured"
                    )

                if form_data.model:
                    set_image_model(request, form_data.model)
                else:
                    set_image_model(request, model)

                data = {
                    "prompt": form_data.prompt,
                    "batch_size": form_data.n,
                    "width": width,
                    "height": height,
                    "steps": steps,
                }

                if form_data.negative_prompt is not None:
                    data["negative_prompt"] = form_data.negative_prompt

                if "cfg_scale" in model_wrapper:
                    data["cfg_scale"] = model_wrapper["cfg_scale"]
                if "sampler" in model_wrapper:
                    data["sampler_name"] = model_wrapper["sampler"]
                if "scheduler" in model_wrapper:
                    data["scheduler"] = model_wrapper["scheduler"]

                r = await asyncio.to_thread(
                    requests.post,
                    url=f"{base_url}/sdapi/v1/txt2img",
                    json=data,
                    headers={"authorization": get_automatic1111_api_key(request)},
                )

                res = r.json()
                log.debug(f"res: {res}")

                images = []

                for image in res["images"]:
                    image_data, content_type = load_b64_image_data(image)
                    url = upload_image(
                        request,
                        {**data, "info": res["info"]},
                        image_data,
                        content_type,
                        user,
                    )
                    images.append({"url": url})
                return images
        except Exception as e:
            log.error(f"Error during image generation: {str(e)}")
            if r is not None:
                try:
                    data = r.json()
                    log.error(f"Response data: {json.dumps(data, indent=2)}")
                    if "error" in data:
                        error = data["error"]["message"]
                    else:
                        error = f"Status {r.status_code}: {r.text}"
                except Exception as json_err:
                    log.error(f"Error parsing response: {str(json_err)}")
                    error = f"Status {r.status_code}: {r.text}"
            else:
                error = str(e)
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES.DEFAULT(error))

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unexpected error in image_generations")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
