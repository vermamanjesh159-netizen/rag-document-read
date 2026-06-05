import os
import time
import asyncio
import shutil
from typing import Dict, Any
from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
import httpx

from database import SessionLocal

router = APIRouter()

# Record startup time for uptime calculation
START_TIME = time.time()

async def check_db() -> Dict[str, Any]:
    """Verify database connectivity and measure latency."""
    db = SessionLocal()
    try:
        start_time = time.time()
        db.execute(text("SELECT 1"))
        latency_ms = (time.time() - start_time) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "error": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": str(e)
        }
    finally:
        db.close()

async def check_groq() -> Dict[str, Any]:
    """Verify Groq API key configuration and active connectivity."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "unconfigured",
            "latency_ms": None,
            "error": "GROQ_API_KEY environment variable is not configured"
        }
        
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            latency_ms = (time.time() - start_time) * 1000
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "latency_ms": round(latency_ms, 2),
                    "error": None
                }
            else:
                return {
                    "status": "unhealthy",
                    "latency_ms": round(latency_ms, 2),
                    "error": f"API returned status code {response.status_code}: {response.text}"
                }
    except httpx.RequestError as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": f"Network connectivity error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": str(e)
        }

async def check_huggingface() -> Dict[str, Any]:
    """Verify Hugging Face API token configuration and endpoint connectivity."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return {
            "status": "unconfigured",
            "latency_ms": None,
            "error": "HUGGINGFACEHUB_API_TOKEN environment variable is not configured"
        }
        
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Connect directly to the model's serverless pipeline endpoint used for embeddings
            response = await client.post(
                "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
                headers={"Authorization": f"Bearer {token}"},
                json={"inputs": "test query for health check"}
            )
            latency_ms = (time.time() - start_time) * 1000
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "latency_ms": round(latency_ms, 2),
                    "error": None
                }
            else:
                return {
                    "status": "unhealthy",
                    "latency_ms": round(latency_ms, 2),
                    "error": f"API returned status code {response.status_code}: {response.text}"
                }
    except httpx.RequestError as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": f"Network connectivity error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "error": str(e)
        }

def check_directories() -> Dict[str, Any]:
    """Verify read/write permissions for essential local directories."""
    results = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for dir_name in ["documents", "vector_stores"]:
        path = os.path.join(base_dir, dir_name)
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        writable = os.access(path, os.W_OK) if exists else False
        
        results[dir_name] = {
            "exists": exists,
            "is_directory": is_dir,
            "writable": writable,
            "status": "healthy" if (exists and is_dir and writable) else "unhealthy"
        }
    return results

def get_system_stats() -> Dict[str, Any]:
    """Collect resource usage statistics from the system."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Disk space
    total, used, free = shutil.disk_usage(base_dir)
    disk_space = {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "used_percent": round((used / total) * 100, 2)
    }

    # CPU load average (Linux/macOS)
    try:
        load1, load5, load15 = os.getloadavg()
        cpu_load = {
            "load_1m": load1,
            "load_5m": load5,
            "load_15m": load15
        }
    except Exception as e:
        cpu_load = {"error": str(e)}

    # Memory usage parsed from /proc/meminfo (reliable for Linux deployments)
    mem_util = {}
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    name = parts[0].strip()
                    val = parts[1].split()[0].strip()
                    meminfo[name] = int(val)
        
        total_mem = meminfo.get("MemTotal", 0)
        free_mem = meminfo.get("MemFree", 0)
        buffers = meminfo.get("Buffers", 0)
        cached = meminfo.get("Cached", 0)
        available_mem = meminfo.get("MemAvailable", free_mem + buffers + cached)
        used_mem = total_mem - available_mem
        mem_util = {
            "total_mb": round(total_mem / 1024, 2),
            "available_mb": round(available_mem / 1024, 2),
            "used_mb": round(used_mem / 1024, 2),
            "used_percent": round((used_mem / total_mem) * 100, 2) if total_mem else 0
        }
    except Exception as e:
        mem_util = {"error": f"Failed to read memory: {str(e)}"}

    # Uptime tracking
    uptime_seconds = time.time() - START_TIME
    days, remaining = divmod(uptime_seconds, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds = divmod(remaining, 60)
    uptime_formatted = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

    return {
        "disk": disk_space,
        "cpu_load": cpu_load,
        "memory": mem_util,
        "uptime": {
            "seconds": round(uptime_seconds, 2),
            "formatted": uptime_formatted
        }
    }

def check_dependencies() -> Dict[str, Any]:
    """Verify standard application package imports and return versions."""
    dependencies = {}
    dep_list = ["fastapi", "sqlalchemy", "langchain", "faiss", "pydantic", "uvicorn"]
    for name in dep_list:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", None)
            if not version:
                if name == "faiss":
                    try:
                        import faiss
                        version = getattr(faiss, "__version__", "available")
                    except Exception:
                        version = "available"
            dependencies[name] = {
                "status": "available",
                "version": version or "unknown"
            }
        except ImportError:
            dependencies[name] = {
                "status": "unavailable",
                "version": None
            }
    return dependencies

@router.get("/api/health")
@router.get("/health")
async def health_check():
    """
    Health Check API.
    
    Verifies database connectivity, directory permissions, system resource stats, 
    key package dependencies, and active connections to external services.
    """
    # Run critical checks concurrently
    db_result, groq_result, hf_result = await asyncio.gather(
        check_db(),
        check_groq(),
        check_huggingface()
    )
    
    dir_results = check_directories()
    sys_stats = get_system_stats()
    dep_results = check_dependencies()
    
    # Assess overall status
    is_db_healthy = db_result["status"] == "healthy"
    # external services can be either configured and healthy, or simply unconfigured (graceful degraded)
    is_groq_healthy = groq_result["status"] in ("healthy", "unconfigured")
    is_hf_healthy = hf_result["status"] in ("healthy", "unconfigured")
    
    directories_healthy = all(d["status"] == "healthy" for d in dir_results.values())
    dependencies_healthy = all(d["status"] == "available" for d in dep_results.values())
    
    # Core health assessment
    if not (is_db_healthy and directories_healthy):
        overall_status = "unhealthy"
    elif not (is_groq_healthy and is_hf_healthy):
        overall_status = "degraded"
    elif groq_result["status"] == "unhealthy" or hf_result["status"] == "unhealthy":
        overall_status = "degraded"
    elif sys_stats["disk"].get("used_percent", 0) > 90.0 or sys_stats["memory"].get("used_percent", 0) > 95.0 or not dependencies_healthy:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
        
    response_payload = {
        "status": overall_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {
            "database": db_result,
            "groq_api": groq_result,
            "huggingface_api": hf_result,
            "storage_directories": {
                "documents": dir_results["documents"]
            },
            "vector_stores": dir_results["vector_stores"]
        },
        "dependencies": dep_results,
        "system_metrics": sys_stats
    }
    
    status_code = status.HTTP_200_OK if overall_status in ("healthy", "degraded") else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=response_payload)
