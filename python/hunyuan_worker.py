from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import struct
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request
import venv
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def configure_text_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_text_streams()
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.pop("TQDM_ASCII", None)


APP_NAME = "Image To 3D"
HUNYUAN_REPO = "Tencent-Hunyuan/Hunyuan3D-2.1"
HUNYUAN_SOURCE_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/archive/refs/heads/main.zip"
WHEEL_SOURCE_URL = "https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1/archive/refs/heads/main.zip"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_REPO = "tencent/Hunyuan3D-2.1"
DINO_REPO = "facebook/dinov2-giant"
TRELLIS_REPO = "microsoft/TRELLIS.2"
TRELLIS_SOURCE_URL = "https://github.com/microsoft/TRELLIS.2.git"
TRELLIS_MODEL_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS_DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
TRELLIS_REMBG_REPO = "briaai/RMBG-2.0"
ENV_MARKER = ".ih3d-env-v4-ready"
HUNYUAN_SOURCE_MARKER = ".ih3d-source-v1-ready"
HUNYUAN_MODEL_MARKER = ".ih3d-models-v1-ready"
TRELLIS_SOURCE_MARKER = ".i3d-trellis-source-v1-ready"
TRELLIS_MODEL_MARKER = ".i3d-trellis-models-v3-ready"
HUNYUAN_ENV_MARKER = ".i3d-hunyuan-env-v2-ready"
TRELLIS_ENV_MARKER = ".i3d-trellis-env-v3-ready"
WHEEL_MARKER = ".ih3d-texture-wheels-v1-ready"
VOXEL_DSS_SIDECAR_MAX_CELLS = 300_000


PBR_CONVERT_UTILS_SOURCE = base64.b64decode(
    "aW1wb3J0IHRyaW1lc2gKaW1wb3J0IHB5Z2x0ZmxpYgppbXBvcnQgbnVtcHkgYXMgbnAKZnJvbSBQ"
    "SUwgaW1wb3J0IEltYWdlCmltcG9ydCBiYXNlNjQKaW1wb3J0IGlvCgoKZGVmIGNvbWJpbmVfbWV0"
    "YWxsaWNfcm91Z2huZXNzKG1ldGFsbGljX3BhdGgsIHJvdWdobmVzc19wYXRoLCBvdXRwdXRfcGF0"
    "aCk6CiAgICAiIiIKICAgIOWwhm1ldGFsbGlj5ZKMcm91Z2huZXNz6LS05Zu+5ZCI5bm25Li65LiA"
    "5byg6LS05Zu+CiAgICBHTELmoLzlvI/opoHmsYJtZXRhbGxpY+WcqELpgJrpgZPvvIxyb3VnaG5l"
    "c3PlnKhH6YCa6YGTCiAgICAiIiIKICAgICMg5Yqg6L296LS05Zu+CiAgICBtZXRhbGxpY19pbWcg"
    "PSBJbWFnZS5vcGVuKG1ldGFsbGljX3BhdGgpLmNvbnZlcnQoIkwiKSAgIyDovazkuLrngbDluqYK"
    "ICAgIHJvdWdobmVzc19pbWcgPSBJbWFnZS5vcGVuKHJvdWdobmVzc19wYXRoKS5jb252ZXJ0KCJM"
    "IikgICMg6L2s5Li654Gw5bqmCgogICAgIyDnoa7kv53lsLrlr7jkuIDoh7QKICAgIGlmIG1ldGFs"
    "bGljX2ltZy5zaXplICE9IHJvdWdobmVzc19pbWcuc2l6ZToKICAgICAgICByb3VnaG5lc3NfaW1n"
    "ID0gcm91Z2huZXNzX2ltZy5yZXNpemUobWV0YWxsaWNfaW1nLnNpemUpCgogICAgIyDliJvlu7pS"
    "R0Llm77lg48KICAgIHdpZHRoLCBoZWlnaHQgPSBtZXRhbGxpY19pbWcuc2l6ZQogICAgY29tYmlu"
    "ZWQgPSBJbWFnZS5uZXcoIlJHQiIsICh3aWR0aCwgaGVpZ2h0KSkKCiAgICAjIOi9rOS4um51bXB5"
    "5pWw57uE5L6/5LqO5pON5L2cCiAgICBtZXRhbGxpY19hcnJheSA9IG5wLmFycmF5KG1ldGFsbGlj"
    "X2ltZykKICAgIHJvdWdobmVzc19hcnJheSA9IG5wLmFycmF5KHJvdWdobmVzc19pbWcpCgogICAg"
    "IyDliJvlu7rlkIjlubbnmoTmlbDnu4QgKFIsIEcsIEIpID0gKEFPLCBSb3VnaG5lc3MsIE1ldGFs"
    "bGljKQogICAgY29tYmluZWRfYXJyYXkgPSBucC56ZXJvcygoaGVpZ2h0LCB3aWR0aCwgMyksIGR0"
    "eXBlPW5wLnVpbnQ4KQogICAgY29tYmluZWRfYXJyYXlbOiwgOiwgMF0gPSAyNTUgICMgUumAmumB"
    "k++8mkFPICjlpoLmnpzmsqHmnIlBT+i0tOWbvu+8jOiuvuS4uueZveiJsikKICAgIGNvbWJpbmVk"
    "X2FycmF5WzosIDosIDFdID0gcm91Z2huZXNzX2FycmF5ICAjIEfpgJrpgZPvvJpSb3VnaG5lc3MK"
    "ICAgIGNvbWJpbmVkX2FycmF5WzosIDosIDJdID0gbWV0YWxsaWNfYXJyYXkgICMgQumAmumBk++8"
    "mk1ldGFsbGljCgogICAgIyDovazlm55QSUzlm77lg4/lubbkv53lrZgKICAgIGNvbWJpbmVkID0g"
    "SW1hZ2UuZnJvbWFycmF5KGNvbWJpbmVkX2FycmF5KQogICAgY29tYmluZWQuc2F2ZShvdXRwdXRf"
    "cGF0aCkKICAgIHJldHVybiBvdXRwdXRfcGF0aAoKCmRlZiBfaW1hZ2VfbWltZV9hbmRfYnl0ZXMo"
    "aW1hZ2VfcGF0aCk6CiAgICAiIiJSZXR1cm4gKG1pbWUsIGJ5dGVzKSBmb3IgYW4gaW1hZ2UsIHJl"
    "LWVuY29kaW5nIHRvIEpQRUcvUE5HL1dFQlAgYXMgbmVlZGVkLiIiIgogICAgd2l0aCBvcGVuKGlt"
    "YWdlX3BhdGgsICJyYiIpIGFzIGY6CiAgICAgICAgZGF0YSA9IGYucmVhZCgpCiAgICBpZiBkYXRh"
    "LnN0YXJ0c3dpdGgoYiJceGZmXHhkOFx4ZmYiKToKICAgICAgICByZXR1cm4gImltYWdlL2pwZWci"
    "LCBkYXRhCiAgICBpZiBkYXRhLnN0YXJ0c3dpdGgoYiJceDg5UE5HXHJcblx4MWFcbiIpOgogICAg"
    "ICAgIHJldHVybiAiaW1hZ2UvcG5nIiwgZGF0YQogICAgaWYgZGF0YS5zdGFydHN3aXRoKGIiUklG"
    "RiIpIGFuZCBkYXRhWzg6MTJdID09IGIiV0VCUCI6CiAgICAgICAgcmV0dXJuICJpbWFnZS93ZWJw"
    "IiwgZGF0YQogICAgIyBVbmtub3duIGZvcm1hdDogcmUtZW5jb2RlIHRvIFBORyB2aWEgUElMIHNv"
    "IHRoZSBNSU1FIGlzIGFsd2F5cyBhY2N1cmF0ZS4KICAgIGltZyA9IEltYWdlLm9wZW4oaW8uQnl0"
    "ZXNJTyhkYXRhKSkuY29udmVydCgiUkdCQSIpCiAgICBidWYgPSBpby5CeXRlc0lPKCkKICAgIGlt"
    "Zy5zYXZlKGJ1ZiwgZm9ybWF0PSJQTkciKQogICAgcmV0dXJuICJpbWFnZS9wbmciLCBidWYuZ2V0"
    "dmFsdWUoKQoKCmRlZiBfZ2xiX2VtYmVkX2ltYWdlKGdsdGYsIGJpbmFyeSwgaW1hZ2VfcGF0aCk6"
    "CiAgICAiIiJBcHBlbmQgaW1hZ2UgYnl0ZXMgdG8gdGhlIEdMQiBiaW5hcnkgYnVmZmVyIGFuZCBy"
    "ZXR1cm4gdGhlIGltYWdlIGluZGV4LgoKICAgIFRleHR1cmVzIGFyZSBlbWJlZGRlZCBhcyBwcm9w"
    "ZXIgR0xCIGJpbmFyeSBidWZmZXJWaWV3IGVudHJpZXMgKHRoZSBzdGFuZGFyZAogICAgd2F5KSBy"
    "YXRoZXIgdGhhbiBiYXNlNjQgZGF0YS1VUklzIGluIHRoZSBKU09OIGNodW5rLiBEYXRhLVVSSSB0"
    "ZXh0dXJlcyBhcmUKICAgIGdvdmVybmVkIGJ5IHRoZSBwYWdlIENvbnRlbnQtU2VjdXJpdHktUG9s"
    "aWN5IGNvbm5lY3Qtc3JjIGFuZCBnZXQgc2lsZW50bHkKICAgIGRyb3BwZWQgaW4gbG9ja2VkLWRv"
    "d24gd2Vidmlld3MsIGxlYXZpbmcgdGhlIG1vZGVsIHVudGV4dHVyZWQuIEJpbmFyeQogICAgYnVm"
    "ZmVyVmlldyB0ZXh0dXJlcyBsb2FkIHRocm91Z2ggdGhlIEdMQiBsb2FkZXIgYW5kIHJlbmRlciBl"
    "dmVyeXdoZXJlLgogICAgIiIiCiAgICBtaW1lLCBkYXRhID0gX2ltYWdlX21pbWVfYW5kX2J5dGVz"
    "KGltYWdlX3BhdGgpCgogICAgIyBUaGUgR0xCIGJpbmFyeSBibG9iIGxpdmVzIGluIGJ1ZmZlciAw"
    "OyBncm93IGl0IHdpdGggNC1ieXRlIGFsaWdubWVudC4KICAgIHBhZCA9ICg0IC0gKGxlbihiaW5h"
    "cnkpICUgNCkpICUgNAogICAgaWYgcGFkOgogICAgICAgIGJpbmFyeSArPSBiIlx4MDAiICogcGFk"
    "CiAgICBvZmZzZXQgPSBsZW4oYmluYXJ5KQogICAgYmluYXJ5ICs9IGRhdGEKICAgIHBhZCA9ICg0"
    "IC0gKGxlbihkYXRhKSAlIDQpKSAlIDQKICAgIGlmIHBhZDoKICAgICAgICBiaW5hcnkgKz0gYiJc"
    "eDAwIiAqIHBhZAoKICAgIGlmIG5vdCBnbHRmLmJ1ZmZlcnM6CiAgICAgICAgZ2x0Zi5idWZmZXJz"
    "ID0gW3B5Z2x0ZmxpYi5CdWZmZXIoYnl0ZUxlbmd0aD0wKV0KICAgIGdsdGYuYnVmZmVyc1swXS5i"
    "eXRlTGVuZ3RoID0gbGVuKGJpbmFyeSkKCiAgICBidWZmZXJfdmlldyA9IHB5Z2x0ZmxpYi5CdWZm"
    "ZXJWaWV3KAogICAgICAgIGJ1ZmZlcj0wLAogICAgICAgIGJ5dGVPZmZzZXQ9b2Zmc2V0LAogICAg"
    "ICAgIGJ5dGVMZW5ndGg9bGVuKGRhdGEpLAogICAgKQogICAgZ2x0Zi5idWZmZXJWaWV3cy5hcHBl"
    "bmQoYnVmZmVyX3ZpZXcpCiAgICBidl9pbmRleCA9IGxlbihnbHRmLmJ1ZmZlclZpZXdzKSAtIDEK"
    "CiAgICBpbWFnZSA9IHB5Z2x0ZmxpYi5JbWFnZShidWZmZXJWaWV3PWJ2X2luZGV4LCBtaW1lVHlw"
    "ZT1taW1lKQogICAgZ2x0Zi5pbWFnZXMuYXBwZW5kKGltYWdlKQogICAgcmV0dXJuIGxlbihnbHRm"
    "LmltYWdlcykgLSAxCgoKZGVmIGNyZWF0ZV9nbGJfd2l0aF9wYnJfbWF0ZXJpYWxzKG9ial9wYXRo"
    "LCB0ZXh0dXJlc19kaWN0LCBvdXRwdXRfcGF0aCk6CiAgICAiIiJDcmVhdGUgYSBHTEIgd2l0aCBm"
    "dWxsIFBCUiBtYXRlcmlhbHMuCgogICAgVGV4dHVyZXMgYXJlIGVtYmVkZGVkIGFzIHN0YW5kYXJk"
    "IEdMQiBiaW5hcnkgYnVmZmVyVmlldyBkYXRhIChjb3JyZWN0IE1JTUUKICAgIHBlciBpbWFnZSks"
    "IG5ldmVyIGFzIGRhdGEtVVJJcywgc28gdGhlIGZpbGUgcmVuZGVycyBpbiBhbnkgR0xCIHZpZXdl"
    "ciAvCiAgICB3ZWJ2aWV3IHJlZ2FyZGxlc3Mgb2YgQ29udGVudC1TZWN1cml0eS1Qb2xpY3kuCgog"
    "ICAgdGV4dHVyZXNfZGljdCA9IHsKICAgICAgICAnYWxiZWRvJzogJ3BhdGgvdG8vYWxiZWRvLnBu"
    "ZycsCiAgICAgICAgJ21ldGFsbGljJzogJ3BhdGgvdG8vbWV0YWxsaWMucG5nJywKICAgICAgICAn"
    "cm91Z2huZXNzJzogJ3BhdGgvdG8vcm91Z2huZXNzLnBuZycsCiAgICAgICAgJ25vcm1hbCc6ICdw"
    "YXRoL3RvL25vcm1hbC5wbmcnLCAgICMgb3B0aW9uYWwKICAgICAgICAnYW8nOiAncGF0aC90by9h"
    "by5wbmcnICAgICAgICAgICAgIyBvcHRpb25hbAogICAgfQogICAgIiIiCiAgICBpbXBvcnQgb3MK"
    "CiAgICAjIFdvcmsgbmV4dCB0byB0aGUgT0JKIHNvIHRlbXBvcmFyeSBmaWxlcyBhcmUgbm90IGxv"
    "c3QgYWNyb3NzIENXRCBjaGFuZ2VzLgogICAgd29ya19kaXIgPSBvcy5wYXRoLmRpcm5hbWUob3Mu"
    "cGF0aC5hYnNwYXRoKG9ial9wYXRoKSkKCiAgICAjIDEuIExvYWQgdGhlIE9CSiBhbmQgZXhwb3J0"
    "IGEgdGVtcG9yYXJ5IEdMQiBjb250YWluaW5nIG9ubHkgZ2VvbWV0cnkuCiAgICBtZXNoID0gdHJp"
    "bWVzaC5sb2FkKG9ial9wYXRoKQogICAgdGVtcF9nbGIgPSBvcy5wYXRoLmpvaW4od29ya19kaXIs"
    "ICJ0ZW1wLmdsYiIpCiAgICBtZXNoLmV4cG9ydCh0ZW1wX2dsYikKCiAgICAjIDIuIExvYWQgaXQg"
    "YmFjayB0byBhdHRhY2ggUEJSIG1hdGVyaWFscy4KICAgIGdsdGYgPSBweWdsdGZsaWIuR0xURjIo"
    "KS5sb2FkKHRlbXBfZ2xiKQoKICAgICMgMy4gQ29tYmluZSBtZXRhbGxpYyArIHJvdWdobmVzcyBp"
    "bnRvIHRoZSBnbFRGIE1SIHRleHR1cmUgKEc9cm91Z2gsIEI9bWV0YWwpLgogICAgaWYgIm1ldGFs"
    "bGljIiBpbiB0ZXh0dXJlc19kaWN0IGFuZCAicm91Z2huZXNzIiBpbiB0ZXh0dXJlc19kaWN0Ogog"
    "ICAgICAgIG1yX2NvbWJpbmVkX3BhdGggPSBvcy5wYXRoLmpvaW4od29ya19kaXIsICJtcl9jb21i"
    "aW5lZC5wbmciKQogICAgICAgIGNvbWJpbmVfbWV0YWxsaWNfcm91Z2huZXNzKAogICAgICAgICAg"
    "ICB0ZXh0dXJlc19kaWN0WyJtZXRhbGxpYyJdLCB0ZXh0dXJlc19kaWN0WyJyb3VnaG5lc3MiXSwg"
    "bXJfY29tYmluZWRfcGF0aAogICAgICAgICkKICAgICAgICB0ZXh0dXJlc19kaWN0WyJtZXRhbGxp"
    "Y1JvdWdobmVzcyJdID0gbXJfY29tYmluZWRfcGF0aAoKICAgICMgNC4gQnVpbGQgYSBjbGVhbiBi"
    "aW5hcnkgYnVmZmVyIGhvbGRpbmcgT05MWSB0aGUgZ2VvbWV0cnkgYWNjZXNzb3IgZGF0YSwKICAg"
    "ICMgICAgdGhlbiBhcHBlbmQgdGhlIFBCUiB0ZXh0dXJlcyBhZnRlciBpdC4gdHJpbWVzaCdzIHRl"
    "bXAgR0xCIG1heSBhbHJlYWR5CiAgICAjICAgIGVtYmVkIGEgKHdoaXRlIGRlZmF1bHQpIGltYWdl"
    "ICsgYnVmZmVyVmlldzsgd2UgZHJvcCB0aG9zZSBhbmQgcHJvdmlkZQogICAgIyAgICBvdXIgb3du"
    "LCBzbyB3ZSBtdXN0IHJlYnVpbGQgdGhlIGJ1ZmZlclZpZXcgdGFibGUgZnJvbSBzY3JhdGNoIGFu"
    "ZAogICAgIyAgICByZS1tYXAgdGhlIGdlb21ldHJ5IGFjY2Vzc29ycyB0byB0aGUgbmV3IGNvbnRp"
    "Z3VvdXMgb2Zmc2V0cy4KICAgIGFjY2Vzc29yX2luZGV4X3RvX2J2ID0ge30KICAgIGZvciBhaSwg"
    "YWNjIGluIGVudW1lcmF0ZShnbHRmLmFjY2Vzc29ycyk6CiAgICAgICAgYnZfaW5kZXggPSBnZXRh"
    "dHRyKGFjYywgImJ1ZmZlclZpZXciLCBOb25lKQogICAgICAgIGlmIGJ2X2luZGV4IGlzIG5vdCBO"
    "b25lOgogICAgICAgICAgICBhY2Nlc3Nvcl9pbmRleF90b19idlthaV0gPSBidl9pbmRleAoKICAg"
    "IG9sZF9idWZmZXJfdmlld3MgPSBnbHRmLmJ1ZmZlclZpZXdzCiAgICBvbGRfYmxvYiA9IGdsdGYu"
    "YmluYXJ5X2Jsb2IoKSBvciBiIiIKCiAgICBiaW5hcnkgPSBieXRlYXJyYXkoKQogICAgbmV3X2J1"
    "ZmZlcl92aWV3cyA9IFtdCiAgICAjIEdlb21ldHJ5IGFjY2Vzc29ycyByZWZlcmVuY2UgYnVmZmVy"
    "Vmlld3MgYnkgaW5kZXg7IGtlZXAgYSBzdGFibGUgbWFwcGluZwogICAgIyBmcm9tIHRoZSBPTEQg"
    "YnVmZmVyVmlldyBpbmRleCB0byB0aGUgTkVXIG9uZSBzbyBhY2Nlc3NvcnMgc3RheSB2YWxpZC4K"
    "ICAgIG9sZF9idl90b19uZXcgPSB7fQogICAgZ2VvbWV0cnlfYnZfaW5kaWNlcyA9IHNldChhY2Nl"
    "c3Nvcl9pbmRleF90b19idi52YWx1ZXMoKSkKICAgIGZvciBvbGRfYnZfaW5kZXggaW4gc29ydGVk"
    "KGdlb21ldHJ5X2J2X2luZGljZXMpOgogICAgICAgIG9sZF9idiA9IG9sZF9idWZmZXJfdmlld3Nb"
    "b2xkX2J2X2luZGV4XQogICAgICAgIGRhdGEgPSBvbGRfYmxvYltvbGRfYnYuYnl0ZU9mZnNldDpv"
    "bGRfYnYuYnl0ZU9mZnNldCArIG9sZF9idi5ieXRlTGVuZ3RoXQogICAgICAgICMgNC1ieXRlIGFs"
    "aWduIHRoZSBydW5uaW5nIGJ1ZmZlciBiZWZvcmUgYXBwZW5kaW5nLgogICAgICAgIHBhZCA9ICg0"
    "IC0gKGxlbihiaW5hcnkpICUgNCkpICUgNAogICAgICAgIGlmIHBhZDoKICAgICAgICAgICAgYmlu"
    "YXJ5ICs9IGIiXHgwMCIgKiBwYWQKICAgICAgICBuZXdfb2Zmc2V0ID0gbGVuKGJpbmFyeSkKICAg"
    "ICAgICBiaW5hcnkgKz0gZGF0YQogICAgICAgIHBhZCA9ICg0IC0gKGxlbihkYXRhKSAlIDQpKSAl"
    "IDQKICAgICAgICBpZiBwYWQ6CiAgICAgICAgICAgIGJpbmFyeSArPSBiIlx4MDAiICogcGFkCiAg"
    "ICAgICAgbmV3X2J2ID0gcHlnbHRmbGliLkJ1ZmZlclZpZXcoCiAgICAgICAgICAgIGJ1ZmZlcj0w"
    "LAogICAgICAgICAgICBieXRlT2Zmc2V0PW5ld19vZmZzZXQsCiAgICAgICAgICAgIGJ5dGVMZW5n"
    "dGg9bGVuKGRhdGEpLAogICAgICAgICkKICAgICAgICBuZXdfaW5kZXggPSBsZW4obmV3X2J1ZmZl"
    "cl92aWV3cykKICAgICAgICBuZXdfYnVmZmVyX3ZpZXdzLmFwcGVuZChuZXdfYnYpCiAgICAgICAg"
    "b2xkX2J2X3RvX25ld1tvbGRfYnZfaW5kZXhdID0gbmV3X2luZGV4CgogICAgIyBSZS1wb2ludCBh"
    "Y2Nlc3NvcnMgYXQgdGhlIHJlbWFwcGVkIGJ1ZmZlclZpZXdzLgogICAgZm9yIGFpLCBhY2MgaW4g"
    "ZW51bWVyYXRlKGdsdGYuYWNjZXNzb3JzKToKICAgICAgICBvbGRfYnYgPSBnZXRhdHRyKGFjYywg"
    "ImJ1ZmZlclZpZXciLCBOb25lKQogICAgICAgIGlmIG9sZF9idiBpcyBub3QgTm9uZSBhbmQgb2xk"
    "X2J2IGluIG9sZF9idl90b19uZXc6CiAgICAgICAgICAgIGFjYy5idWZmZXJWaWV3ID0gb2xkX2J2"
    "X3RvX25ld1tvbGRfYnZdCgogICAgIyBSZXNldCB0aGUgdGFibGVzOiBnZW9tZXRyeSB2aWV3cyBm"
    "aXJzdCwgaW1hZ2VzL3RleHR1cmVzIGFwcGVuZGVkIG5leHQuCiAgICBnbHRmLmJ1ZmZlclZpZXdz"
    "ID0gbmV3X2J1ZmZlcl92aWV3cwogICAgZ2x0Zi5pbWFnZXMgPSBbXQogICAgZ2x0Zi50ZXh0dXJl"
    "cyA9IFtdCgogICAgaWYgZ2x0Zi5idWZmZXJzOgogICAgICAgIGdsdGYuYnVmZmVyc1swXS51cmkg"
    "PSBOb25lCiAgICBlbHNlOgogICAgICAgIGdsdGYuYnVmZmVycyA9IFtweWdsdGZsaWIuQnVmZmVy"
    "KGJ5dGVMZW5ndGg9MCldCgogICAgdGV4dHVyZXMgPSBbXQogICAgdGV4dHVyZV9pbmRleCA9IDAK"
    "CiAgICBkZWYgYWRkX3RleHR1cmUoaW1hZ2VfcGF0aCk6CiAgICAgICAgbm9ubG9jYWwgdGV4dHVy"
    "ZV9pbmRleAogICAgICAgIGltYWdlX2luZGV4ID0gX2dsYl9lbWJlZF9pbWFnZShnbHRmLCBiaW5h"
    "cnksIGltYWdlX3BhdGgpCiAgICAgICAgdGV4dHVyZSA9IHB5Z2x0ZmxpYi5UZXh0dXJlKHNvdXJj"
    "ZT1pbWFnZV9pbmRleCkKICAgICAgICBnbHRmLnRleHR1cmVzLmFwcGVuZCh0ZXh0dXJlKQogICAg"
    "ICAgIGlkeCA9IHRleHR1cmVfaW5kZXgKICAgICAgICB0ZXh0dXJlX2luZGV4ICs9IDEKICAgICAg"
    "ICByZXR1cm4gaWR4CgogICAgIyA1LiBCdWlsZCB0aGUgUEJSIG1hdGVyaWFsLCBhdHRhY2hpbmcg"
    "dGV4dHVyZXMgaW4gZ2xURiBvcmRlci4KICAgIHBicl9tZXRhbGxpY19yb3VnaG5lc3MgPSBweWds"
    "dGZsaWIuUGJyTWV0YWxsaWNSb3VnaG5lc3MoCiAgICAgICAgYmFzZUNvbG9yRmFjdG9yPVsxLjAs"
    "IDEuMCwgMS4wLCAxLjBdLCBtZXRhbGxpY0ZhY3Rvcj0xLjAsIHJvdWdobmVzc0ZhY3Rvcj0xLjAK"
    "ICAgICkKICAgIGlmIHRleHR1cmVzX2RpY3QuZ2V0KCJhbGJlZG8iKToKICAgICAgICBwYnJfbWV0"
    "YWxsaWNfcm91Z2huZXNzLmJhc2VDb2xvclRleHR1cmUgPSBweWdsdGZsaWIuVGV4dHVyZUluZm8o"
    "CiAgICAgICAgICAgIGluZGV4PWFkZF90ZXh0dXJlKHRleHR1cmVzX2RpY3RbImFsYmVkbyJdKQog"
    "ICAgICAgICkKICAgIGlmIHRleHR1cmVzX2RpY3QuZ2V0KCJtZXRhbGxpY1JvdWdobmVzcyIpOgog"
    "ICAgICAgIHBicl9tZXRhbGxpY19yb3VnaG5lc3MubWV0YWxsaWNSb3VnaG5lc3NUZXh0dXJlID0g"
    "cHlnbHRmbGliLlRleHR1cmVJbmZvKAogICAgICAgICAgICBpbmRleD1hZGRfdGV4dHVyZSh0ZXh0"
    "dXJlc19kaWN0WyJtZXRhbGxpY1JvdWdobmVzcyJdKQogICAgICAgICkKCiAgICBtYXRlcmlhbCA9"
    "IHB5Z2x0ZmxpYi5NYXRlcmlhbCgKICAgICAgICBuYW1lPSJQQlJfTWF0ZXJpYWwiLCBwYnJNZXRh"
    "bGxpY1JvdWdobmVzcz1wYnJfbWV0YWxsaWNfcm91Z2huZXNzCiAgICApCgogICAgaWYgdGV4dHVy"
    "ZXNfZGljdC5nZXQoIm5vcm1hbCIpOgogICAgICAgIG1hdGVyaWFsLm5vcm1hbFRleHR1cmUgPSBw"
    "eWdsdGZsaWIuTm9ybWFsVGV4dHVyZUluZm8oCiAgICAgICAgICAgIGluZGV4PWFkZF90ZXh0dXJl"
    "KHRleHR1cmVzX2RpY3RbIm5vcm1hbCJdKQogICAgICAgICkKICAgIGlmIHRleHR1cmVzX2RpY3Qu"
    "Z2V0KCJhbyIpOgogICAgICAgIG1hdGVyaWFsLm9jY2x1c2lvblRleHR1cmUgPSBweWdsdGZsaWIu"
    "T2NjbHVzaW9uVGV4dHVyZUluZm8oCiAgICAgICAgICAgIGluZGV4PWFkZF90ZXh0dXJlKHRleHR1"
    "cmVzX2RpY3RbImFvIl0pCiAgICAgICAgKQoKICAgIGdsdGYudGV4dHVyZXMgPSB0ZXh0dXJlcwog"
    "ICAgZ2x0Zi5tYXRlcmlhbHMgPSBbbWF0ZXJpYWxdCgogICAgIyA2LiBBc3NpZ24gdGhlIG1hdGVy"
    "aWFsIHRvIGV2ZXJ5IHByaW1pdGl2ZS4KICAgIGlmIGdsdGYubWVzaGVzOgogICAgICAgIGZvciBw"
    "cmltaXRpdmUgaW4gZ2x0Zi5tZXNoZXNbMF0ucHJpbWl0aXZlczoKICAgICAgICAgICAgcHJpbWl0"
    "aXZlLm1hdGVyaWFsID0gMAoKICAgICMgNy4gU2F2ZSBhcyBhIHJlYWwgR0xCIChiaW5hcnkgYmxv"
    "YiBpbiB0aGUgZmluYWwgY2h1bmspLgogICAgZ2x0Zi5zZXRfYmluYXJ5X2Jsb2IoYnl0ZXMoYmlu"
    "YXJ5KSkKICAgIGdsdGYuYnVmZmVyc1swXS5ieXRlTGVuZ3RoID0gbGVuKGJpbmFyeSkKICAgIGds"
    "dGYuc2F2ZShvdXRwdXRfcGF0aCkKCiAgICAjIENsZWFuIHVwIHRoZSB0ZW1wIGdlb21ldHJ5IEdM"
    "Qi4KICAgIHRyeToKICAgICAgICBvcy5yZW1vdmUodGVtcF9nbGIpCiAgICBleGNlcHQgT1NFcnJv"
    "cjoKICAgICAgICBwYXNzCgogICAgcHJpbnQoZiJQQlIgR0xC5paH5Lu25bey5L+d5a2YOiB7b3V0"
    "cHV0X3BhdGh9IikKCgo="
).decode("utf-8")
PBR_CONVERT_UTILS_SOURCE = PBR_CONVERT_UTILS_SOURCE.replace(
    "gltf.textures.append(texture)",
    "textures.append(texture)",
)

BOOTSTRAP_PACKAGES = ["pip>=24.3", "setuptools<81", "wheel>=0.43"]
DEFAULT_PYTORCH_PROFILE = "cu124"
TRANSFORMERS_PIN = "transformers==5.12.1"
HUGGINGFACE_HUB_PIN = "huggingface-hub==1.19.0"
TOKENIZERS_PIN = "tokenizers==0.22.2"
PYTORCH_RUNTIME_PROFILES = {
    "cu124": {
        "label": "PyTorch CUDA 12.4",
        "index_url": "https://download.pytorch.org/whl/cu124",
        "packages": [
            "torch==2.5.1+cu124",
            "torchvision==0.20.1+cu124",
            "torchaudio==2.5.1+cu124",
        ],
    },
    "cu128": {
        "label": "PyTorch CUDA 12.8",
        "index_url": "https://download.pytorch.org/whl/cu128",
        "packages": [
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "torchaudio==2.7.1+cu128",
        ],
    },
}
COMPAT_PACKAGES = [
    "numpy==1.26.4",
    "requests>=2.32.0",
    "pillow>=10.4.0",
]
HUNYUAN_RUNTIME_PACKAGES = [
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    TRANSFORMERS_PIN,
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "pytorch-lightning==1.9.5",
    HUGGINGFACE_HUB_PIN,
    TOKENIZERS_PIN,
    "safetensors==0.4.4",
    "scipy==1.14.1",
    "einops==0.8.0",
    "opencv-python==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "rembg==2.0.65",
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "trimesh==4.4.7",
    "pymeshlab==2022.2.post3",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "configargparse==1.7",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "onnxruntime-gpu>=1.20.0",
    "torchmetrics==1.6.0",
    "timm",
    "torchdiffeq",
]
TRELLIS_BASIC_PACKAGES = [
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "ninja",
    "trimesh",
    TRANSFORMERS_PIN,
    HUGGINGFACE_HUB_PIN,
    TOKENIZERS_PIN,
    "tensorboard",
    "pandas",
    "lpips",
    "zstandard",
    "kornia",
    "timm",
    "igraph",
    "pymeshfix",
    "xatlas",
    "pygltflib==1.16.3",
]
TRELLIS_UTILS3D_PACKAGE = "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
TRELLIS_EXTENSION_REPOS = {
    "nvdiffrast": ("https://github.com/NVlabs/nvdiffrast.git", "v0.4.0"),
    "nvdiffrec": ("https://github.com/JeffreyXiang/nvdiffrec.git", "renderutils"),
    "CuMesh": ("https://github.com/JeffreyXiang/CuMesh.git", None),
    "FlexGEMM": ("https://github.com/JeffreyXiang/FlexGEMM.git", None),
}
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply"}
PYTHON_VERSION_RE = re.compile(r"Python\s+(\d+)\.(\d+)\.(\d+)")
MSVC_ENV_CACHE: dict[str, str] | None = None
CURRENT_JOB_MANIFEST: dict[str, Any] | None = None
CURRENT_JOB_MANIFEST_PATH: Path | None = None

TRELLIS_PRESETS: dict[str, dict[str, Any]] = {
    "preview": {
        "trellisResolution": 512,
        "trellisSparseSteps": 12,
        "trellisShapeSteps": 12,
        "trellisTextureSteps": 12,
        "trellisTextureGuidance": 3.0,
        "trellisDecimationTarget": 20_000,
        "trellisTextureSize": 1024,
        "trellisRemesh": False,
    },
    "standard": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 12,
        "trellisShapeSteps": 12,
        "trellisTextureSteps": 12,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 100_000,
        "trellisTextureSize": 2048,
        "trellisRemesh": False,
    },
    "quality": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 25,
        "trellisShapeSteps": 25,
        "trellisTextureSteps": 25,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 200_000,
        "trellisTextureSize": 2048,
        "trellisRemesh": False,
    },
    "highpoly": {
        "trellisResolution": 1024,
        "trellisSparseSteps": 25,
        "trellisShapeSteps": 25,
        "trellisTextureSteps": 25,
        "trellisTextureGuidance": 7.5,
        "trellisDecimationTarget": 500_000,
        "trellisTextureSize": 4096,
        "trellisRemesh": False,
    },
}


def normalize_pytorch_profile(value: Any) -> str:
    profile = str(value or DEFAULT_PYTORCH_PROFILE).strip()
    return profile if profile in PYTORCH_RUNTIME_PROFILES else DEFAULT_PYTORCH_PROFILE


def env_marker_profile(marker: Path) -> str | None:
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if text == "ok":
        return "cu124"
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "pytorch":
            return normalize_pytorch_profile(value)
    return None


def env_marker_value(marker: Path, wanted_key: str) -> str | None:
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == wanted_key:
            return value.strip()
    return None


def env_marker_flavor(marker: Path) -> str | None:
    value = env_marker_value(marker, "runtime")
    if value:
        return value
    if marker.exists():
        return "legacy-full"
    return None


def normalize_model_backend(value: Any) -> str:
    backend = str(value or "hunyuan").strip()
    if backend in {"trellis", "trellis2"}:
        return "trellis2"
    return "hunyuan"


def apply_trellis_preset(options: dict[str, Any]) -> None:
    preset_name = str(options.get("trellisPreset", "standard")).strip().lower()
    if preset_name == "custom":
        return
    preset = TRELLIS_PRESETS.get(preset_name)
    if preset is None:
        options["trellisPreset"] = "custom"
        return
    options.update(preset)


def env_dir_for_profile(app_data_dir: Path, pytorch_profile: Any = None) -> Path:
    profile = normalize_pytorch_profile(pytorch_profile)
    profiled_env = app_data_dir / f"python-env-{profile}"
    if profiled_env.exists():
        return profiled_env

    legacy_env = app_data_dir / "python-env"
    if legacy_env.exists() and env_marker_profile(legacy_env / ENV_MARKER) == profile:
        return legacy_env

    return profiled_env


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def recover_interrupted_manifests(history_dir: Path) -> None:
    if not history_dir.is_dir():
        return
    for manifest_path in history_dir.glob("*.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") not in {"running", "canceling"}:
                continue
            manifest["status"] = "interrupted"
            manifest["message"] = "The previous worker stopped before this job completed."
            manifest["endedAt"] = utc_now_iso()
            started_epoch = manifest.get("_startedEpoch")
            if isinstance(started_epoch, (int, float)):
                manifest["elapsedSeconds"] = round(max(0.0, time.time() - started_epoch), 3)
            write_json_atomic(manifest_path, manifest)
        except Exception:
            continue


def begin_job_manifest(job: dict[str, Any], paths: dict[str, Path], backend: str) -> None:
    global CURRENT_JOB_MANIFEST, CURRENT_JOB_MANIFEST_PATH
    options = dict(job.get("options", {}))
    history_dir = paths["jobs_dir"] / "history"
    recover_interrupted_manifests(history_dir)
    job_id = str(job.get("job_id") or f"job-{int(time.time())}")
    preset = (
        str(options.get("trellisPreset", "custom"))
        if backend == "trellis2"
        else f"{options.get('mode', 'meshTexture')}:{options.get('meshDetail', 'custom')}"
    )
    started_epoch = time.time()
    CURRENT_JOB_MANIFEST_PATH = history_dir / f"{job_id}.json"
    CURRENT_JOB_MANIFEST = {
        "schema": "image-to-3d.job.v1",
        "id": job_id,
        "action": str(job.get("action", "generate")),
        "backend": backend,
        "preset": preset,
        "status": "running",
        "message": "Job started.",
        "startedAt": utc_now_iso(),
        "endedAt": None,
        "elapsedSeconds": None,
        "_startedEpoch": started_epoch,
        "inputs": [str(value) for value in job.get("paths", [])],
        "outputs": [],
        "failedCount": None,
        "totalCount": len(job.get("paths", [])),
        "options": options,
        "items": [],
        "stages": [],
    }
    write_json_atomic(CURRENT_JOB_MANIFEST_PATH, CURRENT_JOB_MANIFEST)


def manifest_item(path: str) -> dict[str, Any]:
    assert CURRENT_JOB_MANIFEST is not None
    items = CURRENT_JOB_MANIFEST.setdefault("items", [])
    for item in items:
        if item.get("path") == path:
            return item
    item = {"path": path, "status": "pending", "message": "", "outputs": []}
    items.append(item)
    return item


def record_job_event(event: dict[str, Any]) -> None:
    if CURRENT_JOB_MANIFEST is None or CURRENT_JOB_MANIFEST_PATH is None:
        return
    try:
        event_type = str(event.get("type", ""))
        message = event.get("message")
        if isinstance(message, str) and message:
            CURRENT_JOB_MANIFEST["message"] = message
        path = event.get("path")
        if isinstance(path, str) and path:
            item = manifest_item(path)
            if event_type == "image_start":
                item["status"] = "running"
                item["startedAt"] = utc_now_iso()
            elif event_type == "image_done":
                item["status"] = "done"
                item["endedAt"] = utc_now_iso()
                item["message"] = message or "Saved"
                item["elapsedSeconds"] = event.get("elapsedSeconds")
                item["outputs"] = list(event.get("outputs") or [])
                if isinstance(event.get("stats"), dict):
                    item["stats"] = event["stats"]
            elif event_type == "image_error":
                item["status"] = "error"
                item["endedAt"] = utc_now_iso()
                item["message"] = message or "Generation failed"
                item["elapsedSeconds"] = event.get("elapsedSeconds")

        if event_type == "stage_timing":
            CURRENT_JOB_MANIFEST.setdefault("stages", []).append(
                {
                    "stage": event.get("stage"),
                    "path": event.get("path"),
                    "elapsedSeconds": event.get("elapsedSeconds"),
                    "recordedAt": utc_now_iso(),
                }
            )
        outputs = event.get("outputs")
        if isinstance(outputs, list):
            known_outputs = CURRENT_JOB_MANIFEST.setdefault("outputs", [])
            for output in outputs:
                if isinstance(output, str) and output not in known_outputs:
                    known_outputs.append(output)
        if isinstance(event.get("failedCount"), int):
            CURRENT_JOB_MANIFEST["failedCount"] = event["failedCount"]
        if isinstance(event.get("totalCount"), int):
            CURRENT_JOB_MANIFEST["totalCount"] = event["totalCount"]

        terminal = event_type in {"done", "error", "canceled"}
        if event_type == "done":
            CURRENT_JOB_MANIFEST["status"] = (
                "error" if int(CURRENT_JOB_MANIFEST.get("failedCount") or 0) > 0 else "done"
            )
        elif event_type in {"error", "canceled"}:
            CURRENT_JOB_MANIFEST["status"] = event_type
        if terminal:
            CURRENT_JOB_MANIFEST["endedAt"] = utc_now_iso()
            started_epoch = CURRENT_JOB_MANIFEST.get("_startedEpoch")
            if isinstance(started_epoch, (int, float)):
                CURRENT_JOB_MANIFEST["elapsedSeconds"] = round(
                    max(0.0, time.time() - started_epoch),
                    3,
                )
        write_json_atomic(CURRENT_JOB_MANIFEST_PATH, CURRENT_JOB_MANIFEST)
    except Exception:
        # History must never make generation fail.
        return


def emit(event_type: str, message: str | None = None, **payload: Any) -> None:
    event: dict[str, Any] = {"type": event_type}
    if message:
        event["message"] = message
    event.update(payload)
    record_job_event(event)
    print(json.dumps(event, ensure_ascii=True), flush=True)


def emit_timing(stage: str, started: float, path: str | None = None) -> None:
    elapsed_seconds = time.perf_counter() - started
    payload: dict[str, Any] = {
        "stage": stage,
        "elapsedSeconds": round(elapsed_seconds, 3),
    }
    if path:
        payload["path"] = path
    emit("stage_timing", f"{stage} finished in {format_elapsed(elapsed_seconds)}", **payload)


def app_paths(app_data_dir: Path, pytorch_profile: Any = None) -> dict[str, Path]:
    # The venv, runtime/source trees, and models may all live in the shared
    # NekoLegendsAI store (so they persist across app version upgrades). The Rust
    # host sets I3D_ENV_DIR / I3D_RUNTIME_DIR / I3D_MODEL_DIR when it has resolved
    # store paths; honor them, falling back to the app-data-relative defaults.
    configured_env_dir = os.environ.get("I3D_ENV_DIR")
    env_dir = (
        Path(configured_env_dir)
        if configured_env_dir
        else env_dir_for_profile(app_data_dir, pytorch_profile)
    )
    configured_runtime_dir = os.environ.get("I3D_RUNTIME_DIR")
    runtime_dir = Path(configured_runtime_dir) if configured_runtime_dir else app_data_dir / "runtime"
    configured_model_dir = os.environ.get("I3D_MODEL_DIR") or os.environ.get("IH3D_MODEL_DIR")
    model_dir = Path(configured_model_dir) if configured_model_dir else app_data_dir / "models"
    return {
        "app_data": app_data_dir,
        "cache_dir": app_data_dir / "cache",
        "tmp_dir": app_data_dir / "tmp",
        "env_dir": env_dir,
        "runtime_dir": runtime_dir,
        "source_dir": runtime_dir / "Hunyuan3D-2.1",
        "hunyuan_source_dir": runtime_dir / "Hunyuan3D-2.1",
        "trellis_source_dir": runtime_dir / "TRELLIS.2",
        "trellis_extensions_dir": runtime_dir / "trellis-extensions",
        "wheel_source_dir": runtime_dir / "ComfyUI-Hunyuan3d-2-1-wheels",
        "model_dir": model_dir,
        "output_dir": app_data_dir / "outputs",
        "jobs_dir": app_data_dir / "jobs",
    }


def configure_portable_cache_env(paths: dict[str, Path], env: dict[str, str] | None = None) -> dict[str, str]:
    target = env if env is not None else os.environ
    cache_dir = paths["cache_dir"]
    tmp_dir = paths["tmp_dir"]
    pip_cache = cache_dir / "pip"
    hf_home = paths["model_dir"] / "huggingface"
    hy3dgen_cache = paths["model_dir"] / "hy3dgen"
    matplotlib_cache = cache_dir / "matplotlib"
    flex_gemm_cache = cache_dir / "flex_gemm"
    triton_cache = cache_dir / "triton"
    for folder in (cache_dir, tmp_dir, pip_cache, hf_home, hy3dgen_cache, matplotlib_cache, flex_gemm_cache, triton_cache):
        folder.mkdir(parents=True, exist_ok=True)
    target["PIP_CACHE_DIR"] = str(pip_cache)
    target["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    target["XDG_CACHE_HOME"] = str(cache_dir)
    target["HF_HOME"] = str(hf_home)
    target["HF_HUB_CACHE"] = str(hf_home / "hub")
    target["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    target["HY3DGEN_MODELS"] = str(hy3dgen_cache)
    target["MPLCONFIGDIR"] = str(matplotlib_cache)
    target["FLEX_GEMM_CACHE_DIR"] = str(flex_gemm_cache)
    target["TRITON_CACHE_DIR"] = str(triton_cache)
    target["TMPDIR"] = str(tmp_dir)
    target["TEMP"] = str(tmp_dir)
    target["TMP"] = str(tmp_dir)
    return target


def venv_python(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def parse_python_version(text: str) -> tuple[int, int, int] | None:
    match = PYTHON_VERSION_RE.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def python_version(python_path: Path) -> tuple[int, int, int] | None:
    try:
        output = subprocess.check_output(
            [str(python_path), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None
    return parse_python_version(output)


def python_can_import(python_path: Path, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def supported_python_version(version: tuple[int, int, int] | None) -> bool:
    return bool(version and version[0] == 3 and 10 <= version[1] <= 12)


def parse_compute_cap(value: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def parse_memory_mb(value: str) -> int:
    match = re.search(r"(\d+)", value.replace(",", ""))
    return int(match.group(1)) if match else 0


def nvidia_gpus() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        cap = parse_compute_cap(parts[2])
        if cap is None:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "capability": cap,
                "memory_mb": parse_memory_mb(parts[3]),
            }
        )
    return gpus


def cap_label(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def torch_supported_cuda_caps() -> list[tuple[int, int]]:
    try:
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json, torch; print(json.dumps(torch.cuda.get_arch_list()))",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        arches = json.loads(output)
    except Exception:
        return []
    caps = []
    for arch in arches:
        if not arch.startswith("sm_"):
            continue
        digits = arch.removeprefix("sm_")
        if len(digits) < 2 or not digits.isdigit():
            continue
        caps.append((int(digits[:-1]), int(digits[-1])))
    return caps


def cuda_cap_is_supported(device_cap: tuple[int, int], supported_caps: list[tuple[int, int]]) -> bool:
    if device_cap in supported_caps:
        return True
    device_major, device_minor = device_cap
    return any(major == device_major and minor <= device_minor for major, minor in supported_caps)


def normalize_cuda_device(value: Any) -> str:
    device = str(value or "auto").strip()
    return device if device else "auto"


def configure_cuda_device(cuda_device: Any = "auto") -> None:
    preferred_device = normalize_cuda_device(cuda_device)

    gpus = nvidia_gpus()
    if not gpus:
        return

    supported_caps = torch_supported_cuda_caps()
    if not supported_caps:
        return

    if preferred_device != "auto":
        selected = next((gpu for gpu in gpus if gpu["index"] == preferred_device), None)
        if selected is None:
            detected = ", ".join(f"{gpu['index']}: {gpu['name']}" for gpu in gpus)
            raise RuntimeError(f"Selected CUDA GPU {preferred_device} was not found. Detected GPUs: {detected}.")
        if not cuda_cap_is_supported(selected["capability"], supported_caps):
            supported = ", ".join(cap_label(cap) for cap in supported_caps)
            raise RuntimeError(
                f"Selected GPU {selected['name']} {cap_label(selected['capability'])} is not supported by this "
                f"PyTorch runtime. Supported by this runtime: {supported}. Choose another GPU or reinstall with "
                "the CUDA 12.8 / RTX 50 runtime profile."
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected["index"]
        emit("setup", f"Using selected CUDA GPU {selected['index']}: {selected['name']}")
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        emit("setup", f"Using CUDA device override: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    default_gpu = gpus[0]
    if cuda_cap_is_supported(default_gpu["capability"], supported_caps):
        emit("setup", f"Using {default_gpu['name']} for CUDA")
        return

    compatible = [gpu for gpu in gpus if cuda_cap_is_supported(gpu["capability"], supported_caps)]
    if not compatible:
        supported = ", ".join(cap_label(cap) for cap in supported_caps)
        detected = ", ".join(f"{gpu['name']} {cap_label(gpu['capability'])}" for gpu in gpus)
        raise RuntimeError(
            "The installed PyTorch CUDA runtime does not support the detected NVIDIA GPU. "
            f"Detected: {detected}. Supported by this runtime: {supported}."
        )

    selected = max(compatible, key=lambda gpu: gpu["memory_mb"])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["index"]
    emit(
        "setup",
        f"Using {selected['name']} for CUDA because {default_gpu['name']} is not supported by this PyTorch runtime",
    )


def sanitize_installer_line(line: str) -> str | None:
    if line.startswith("Collecting "):
        package = line.removeprefix("Collecting ").split()[0]
        return f"Resolving dependency: {package}"
    if line.startswith("Downloading "):
        return "Downloading dependency packages"
    if line.startswith("Installing collected packages"):
        return "Installing dependency packages"
    if line.startswith("Successfully installed"):
        return "Dependency install step finished"
    if "already satisfied" in line.lower():
        return None
    if line.startswith("ERROR:") or " error:" in line.lower() or "failed" in line.lower():
        return line
    return None


def run_checked(command: list[str], message: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    emit("setup", f"{message}. This can take several minutes.")
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    tail: list[str] = []
    error_lines: list[str] = []
    last_status: str | None = None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        tail = tail[-80:]
        lowered = line.lower()
        if line.startswith("ERROR:") or " error:" in lowered or "failed" in lowered:
            error_lines.append(line)
            error_lines = error_lines[-20:]
        status = sanitize_installer_line(line)
        if status and status != last_status:
            emit("setup", status)
            last_status = status
    code = process.wait()
    if code != 0:
        sections = []
        if error_lines:
            sections.append("Error lines:\n" + "\n".join(error_lines))
        sections.append("Last installer output:\n" + ("\n".join(tail) if tail else "No installer output was captured."))
        details = "\n\n".join(sections)
        raise RuntimeError(f"{message} failed.\n\n{details}")


def emit_download_progress(name: str, downloaded: int, total: int | None, message: str | None = None) -> None:
    percent = None
    if total and total > 0:
        percent = min(100.0, round(downloaded / total * 100, 1))
    emit(
        "download_progress",
        message or f"Downloading {name}",
        name=name,
        bytesDownloaded=downloaded,
        bytesTotal=total,
        percent=percent,
    )


def emit_item_progress(name: str, current: int, total: int | None, message: str | None = None) -> None:
    percent = None
    if total and total > 0:
        percent = min(100.0, round(current / total * 100, 1))
    emit(
        "download_progress",
        message or f"Downloading {name}",
        name=name,
        itemsDownloaded=current,
        itemsTotal=total,
        percent=percent,
    )


class SilentProgressWriter:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def hf_snapshot_progress_class(label: str):
    from tqdm.auto import tqdm

    status_emitted = False

    class HfSnapshotProgress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal status_emitted
            kwargs["file"] = SilentProgressWriter()
            kwargs["ascii"] = True
            kwargs["dynamic_ncols"] = False
            kwargs.setdefault("mininterval", 0.5)
            super().__init__(*args, **kwargs)
            if not status_emitted:
                emit(
                    "download_progress",
                    f"Downloading {label}. Please wait...",
                    name=label,
                )
                status_emitted = True

    return HfSnapshotProgress


def download_file(url: str, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ImageTo3D/26.7.29"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            temp_path = destination.with_suffix(destination.suffix + ".tmp")
            downloaded = 0
            next_emit = 0
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_emit:
                        emit_download_progress(label, downloaded, total)
                        next_emit = downloaded + max((total or 100_000_000) // 100, 8 * 1024 * 1024)
            temp_path.replace(destination)
            emit_download_progress(label, downloaded, total or downloaded, f"Downloaded {label}")
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not download {label}: {error}") from error


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                normalized = Path(info.filename)
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise RuntimeError(f"Refusing unsafe zip entry: {info.filename}")
            archive.extractall(temp_dir)
        roots = [path for path in temp_dir.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Unexpected archive shape in {zip_path}")
        shutil.move(str(roots[0]), str(destination))
    return destination


def ensure_hunyuan_source(paths: dict[str, Path]) -> Path:
    source_dir = paths["hunyuan_source_dir"]
    marker = source_dir / HUNYUAN_SOURCE_MARKER
    if marker.exists():
        return source_dir

    emit("setup", f"Downloading Hunyuan3D source: {HUNYUAN_REPO}")
    archive = paths["runtime_dir"] / "Hunyuan3D-2.1-main.zip"
    download_file(HUNYUAN_SOURCE_URL, archive, "Hunyuan3D source")
    safe_extract_zip(archive, source_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return source_dir


def clone_or_update(repo_url: str, destination: Path, label: str, branch_or_tag: str | None = None, recursive: bool = False) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(f"Git is required to install {label}. Install Git and run setup again.")
    if destination.is_dir() and (destination / ".git").is_dir():
        run_checked([git, "-C", str(destination), "fetch", "--depth", "1"], f"Refreshing {label} source")
        if branch_or_tag:
            run_checked([git, "-C", str(destination), "checkout", branch_or_tag], f"Selecting {label} revision")
        return
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [git, "clone", "--depth", "1"]
    if recursive:
        command.append("--recursive")
    if branch_or_tag:
        command.extend(["--branch", branch_or_tag])
    command.extend([repo_url, str(destination)])
    run_checked(command, f"Downloading {label} source")


def patch_trellis_dinov3_source(source_dir: Path) -> None:
    target_files = [
        source_dir / "trellis2" / "modules" / "image_feature_extractor.py",
        source_dir / "trellis2" / "trainers" / "flow_matching" / "mixins" / "image_conditioned.py",
    ]
    old = "        for i, layer_module in enumerate(self.model.layer):\n"
    new = (
        "        # Current Transformers exposes DINOv3 blocks under model.layer.\n"
        "        transformer_layers = getattr(self.model, \"layer\", None)\n"
        "        if transformer_layers is None and hasattr(self.model, \"model\"):\n"
        "            transformer_layers = getattr(self.model.model, \"layer\", None)\n"
        "        if transformer_layers is None:\n"
        "            raise AttributeError(\"DINOv3ViTModel exposes neither layer nor model.layer\")\n"
        "        for i, layer_module in enumerate(transformer_layers):\n"
    )
    changed = False
    for file_path in target_files:
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8-sig")
        if "transformer_layers = getattr(self.model, \"layer\", None)" in text:
            continue
        patched = text.replace(old, new)
        if patched != text:
            file_path.write_text(patched, encoding="utf-8", newline="\n")
            changed = True
    if changed:
        emit("setup", "Patched TRELLIS.2 DINOv3 feature extractor for current Transformers")


def patch_trellis_attention_source(source_dir: Path) -> None:
    changed = False

    sparse_config = source_dir / "trellis2" / "modules" / "sparse" / "config.py"
    if sparse_config.is_file():
        text = sparse_config.read_text(encoding="utf-8")
        patched = text.replace(
            "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
            "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']",
        ).replace(
            "def set_attn_backend(backend: Literal['xformers', 'flash_attn']):",
            "def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']):",
        )
        if patched != text:
            sparse_config.write_text(patched, encoding="utf-8", newline="\n")
            changed = True

    sparse_full_attn = source_dir / "trellis2" / "modules" / "sparse" / "attention" / "full_attn.py"
    if sparse_full_attn.is_file():
        text = sparse_full_attn.read_text(encoding="utf-8")
        patched = text
        if "def _sdpa_by_sequence(" not in patched:
            patched = patched.replace(
                "__all__ = [\n"
                "    'sparse_scaled_dot_product_attention',\n"
                "]\n",
                "__all__ = [\n"
                "    'sparse_scaled_dot_product_attention',\n"
                "]\n"
                "\n"
                "\n"
                "def _sdpa_by_sequence(q, k, v, q_seqlen, kv_seqlen):\n"
                "    from torch.nn.functional import scaled_dot_product_attention as sdpa\n"
                "\n"
                "    outputs = []\n"
                "    q_start = 0\n"
                "    kv_start = 0\n"
                "    for q_len, kv_len in zip(q_seqlen, kv_seqlen):\n"
                "        q_len = int(q_len)\n"
                "        kv_len = int(kv_len)\n"
                "        q_chunk = q[q_start:q_start + q_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        k_chunk = k[kv_start:kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        v_chunk = v[kv_start:kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        out = sdpa(q_chunk, k_chunk, v_chunk).squeeze(0).permute(1, 0, 2)\n"
                "        outputs.append(out)\n"
                "        q_start += q_len\n"
                "        kv_start += kv_len\n"
                "    if outputs:\n"
                "        return torch.cat(outputs, dim=0)\n"
                "    return q.new_empty((0, q.shape[1], v.shape[-1]))\n",
            )
        if "elif config.ATTN == 'sdpa':\n        if num_all_args == 1:" not in patched:
            patched = patched.replace(
                "    elif config.ATTN == 'flash_attn':\n",
                "    elif config.ATTN == 'sdpa':\n"
                "        if num_all_args == 1:\n"
                "            q, k, v = qkv.unbind(dim=1)\n"
                "        elif num_all_args == 2:\n"
                "            k, v = kv.unbind(dim=1)\n"
                "        out = _sdpa_by_sequence(q, k, v, q_seqlen, kv_seqlen)\n"
                "    elif config.ATTN == 'flash_attn':\n",
            )
        if patched != text:
            sparse_full_attn.write_text(patched, encoding="utf-8", newline="\n")
            changed = True

    sparse_windowed_attn = source_dir / "trellis2" / "modules" / "sparse" / "attention" / "windowed_attn.py"
    if sparse_windowed_attn.is_file():
        text = sparse_windowed_attn.read_text(encoding="utf-8")
        patched = text
        if "def _sdpa_windowed(" not in patched:
            patched = patched.replace(
                "__all__ = [\n"
                "    'sparse_windowed_scaled_dot_product_self_attention',\n"
                "    'sparse_windowed_scaled_dot_product_cross_attention',\n"
                "]\n",
                "__all__ = [\n"
                "    'sparse_windowed_scaled_dot_product_self_attention',\n"
                "    'sparse_windowed_scaled_dot_product_cross_attention',\n"
                "]\n"
                "\n"
                "\n"
                "def _sdpa_windowed(q, k, v, q_seq_lens, kv_seq_lens):\n"
                "    from torch.nn.functional import scaled_dot_product_attention as sdpa\n"
                "\n"
                "    outputs = []\n"
                "    q_start = 0\n"
                "    kv_start = 0\n"
                "    for q_len, kv_len in zip(q_seq_lens, kv_seq_lens):\n"
                "        q_len = int(q_len)\n"
                "        kv_len = int(kv_len)\n"
                "        q_chunk = q[q_start:q_start + q_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        k_chunk = k[kv_start:kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        v_chunk = v[kv_start:kv_start + kv_len].permute(1, 0, 2).unsqueeze(0)\n"
                "        out = sdpa(q_chunk, k_chunk, v_chunk).squeeze(0).permute(1, 0, 2)\n"
                "        outputs.append(out)\n"
                "        q_start += q_len\n"
                "        kv_start += kv_len\n"
                "    if outputs:\n"
                "        return torch.cat(outputs, dim=0)\n"
                "    return q.new_empty((0, q.shape[1], v.shape[-1]))\n",
            )
        if "elif config.ATTN == 'sdpa':\n        attn_func_args = {}" not in patched:
            patched = patched.replace(
                "    elif config.ATTN == 'flash_attn':\n"
                "        attn_func_args = {\n"
                "            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),\n"
                "            'max_seqlen': torch.max(seq_lens)\n"
                "        }\n",
                "    elif config.ATTN == 'flash_attn':\n"
                "        attn_func_args = {\n"
                "            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),\n"
                "            'max_seqlen': torch.max(seq_lens)\n"
                "        }\n"
                "    elif config.ATTN == 'sdpa':\n"
                "        attn_func_args = {}\n",
            )
        if "elif config.ATTN == 'sdpa':\n        q, k, v = qkv_feats.unbind(dim=1)" not in patched:
            patched = patched.replace(
                "    elif config.ATTN == 'flash_attn':\n"
                "        if 'flash_attn' not in globals():\n"
                "            import flash_attn\n"
                "        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]\n",
                "    elif config.ATTN == 'flash_attn':\n"
                "        if 'flash_attn' not in globals():\n"
                "            import flash_attn\n"
                "        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]\n"
                "    elif config.ATTN == 'sdpa':\n"
                "        q, k, v = qkv_feats.unbind(dim=1)\n"
                "        out = _sdpa_windowed(q, k, v, seq_lens, seq_lens)\n",
                1,
            )
        if "elif config.ATTN == 'sdpa':\n        k, v = kv_feats.unbind(dim=1)" not in patched:
            patched = patched.replace(
                "    elif config.ATTN == 'flash_attn':\n"
                "        if 'flash_attn' not in globals():\n"
                "            import flash_attn\n"
                "        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,\n"
                "            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],\n"
                "            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],\n"
                "        )  # [M, H, C]\n",
                "    elif config.ATTN == 'flash_attn':\n"
                "        if 'flash_attn' not in globals():\n"
                "            import flash_attn\n"
                "        out = flash_attn.flash_attn_varlen_kvpacked_func(q_feats, kv_feats,\n"
                "            cu_seqlens_q=q_attn_func_args['cu_seqlens'], cu_seqlens_k=kv_attn_func_args['cu_seqlens'],\n"
                "            max_seqlen_q=q_attn_func_args['max_seqlen'], max_seqlen_k=kv_attn_func_args['max_seqlen'],\n"
                "        )  # [M, H, C]\n"
                "    elif config.ATTN == 'sdpa':\n"
                "        k, v = kv_feats.unbind(dim=1)\n"
                "        out = _sdpa_windowed(q_feats, k, v, q_seq_lens, kv_seq_lens)\n",
                1,
            )
        if patched != text:
            sparse_windowed_attn.write_text(patched, encoding="utf-8", newline="\n")
            changed = True

    if changed:
        emit("setup", "Patched TRELLIS.2 sparse attention to support PyTorch SDPA")


def ensure_trellis_source(paths: dict[str, Path]) -> Path:
    source_dir = paths["trellis_source_dir"]
    marker = source_dir / TRELLIS_SOURCE_MARKER
    if marker.exists() and (source_dir / "trellis2").is_dir():
        patch_trellis_dinov3_source(source_dir)
        patch_trellis_attention_source(source_dir)
        return source_dir
    clone_or_update(TRELLIS_SOURCE_URL, source_dir, TRELLIS_REPO, recursive=True)
    if not (source_dir / "trellis2").is_dir():
        raise RuntimeError("TRELLIS.2 source did not contain the trellis2 Python package.")
    patch_trellis_dinov3_source(source_dir)
    patch_trellis_attention_source(source_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return source_dir


def patch_hunyuan_source(source_dir: Path) -> None:
    rembg_source = source_dir / "hy3dshape" / "hy3dshape" / "rembg.py"
    if rembg_source.is_file():
        content = rembg_source.read_text(encoding="utf-8")
        original = content
        if "def preferred_onnx_providers" not in content:
            content = content.replace(
                "from rembg import remove, new_session\n",
                "from rembg import remove, new_session\n"
                "\n"
                "\n"
                "def preferred_onnx_providers():\n"
                "    try:\n"
                "        import onnxruntime as ort\n"
                "        available = ort.get_available_providers()\n"
                "    except Exception:\n"
                "        return None\n"
                "    if 'CUDAExecutionProvider' in available:\n"
                "        return ['CUDAExecutionProvider', 'CPUExecutionProvider']\n"
                "    return None\n",
            )
        if "providers = preferred_onnx_providers()" not in content:
            content = content.replace(
                "    def __init__(self):\n        self.session = new_session()",
                "    def __init__(self):\n"
                "        providers = preferred_onnx_providers()\n"
                "        if providers:\n"
                "            try:\n"
                "                self.session = new_session(providers=providers)\n"
                "                return\n"
                "            except Exception:\n"
                "                pass\n"
                "        self.session = new_session()",
            )
        if content != original:
            rembg_source.write_text(content, encoding="utf-8")
            emit("setup", "Patched background remover to prefer ONNX CUDA")

    image_super_utils = source_dir / "hy3dpaint" / "utils" / "image_super_utils.py"
    if image_super_utils.is_file():
        content = image_super_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "import numpy as np\nfrom PIL import Image",
            "import numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom PIL import Image",
        )
        content = content.replace(
            "import numpy as np\nimport torch\nfrom PIL import Image",
            "import numpy as np\nimport torch\nfrom torch.nn import functional as F\nfrom PIL import Image",
        )
        if "def enhance_batch" not in content:
            content = content.replace(
                "    def __call__(self, image):\n"
                "        output, _ = self.upsampler.enhance(np.array(image))\n"
                "        output = Image.fromarray(output)\n"
                "        return output",
                "    def __call__(self, image):\n"
                "        output, _ = self.upsampler.enhance(np.array(image))\n"
                "        output = Image.fromarray(output)\n"
                "        return output\n"
                "\n"
                "    def enhance_batch(self, images):\n"
                "        if len(images) <= 1:\n"
                "            return [self(image) for image in images]\n"
                "        try:\n"
                "            if self.upsampler.tile_size > 0:\n"
                "                return [self(image) for image in images]\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32)\n"
                "                if len(array.shape) != 3 or array.shape[2] != 3:\n"
                "                    return [self(image) for image in images]\n"
                "                max_range = 65535 if np.max(array) > 256 else 255\n"
                "                array = array / max_range\n"
                "                array = array[:, :, [2, 1, 0]]\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            pre_pad = getattr(self.upsampler, \"pre_pad\", 0)\n"
                "            if pre_pad != 0:\n"
                "                batch = F.pad(batch, (0, pre_pad, 0, pre_pad), \"reflect\")\n"
                "            mod_scale = None\n"
                "            if self.upsampler.scale == 2:\n"
                "                mod_scale = 2\n"
                "            elif self.upsampler.scale == 1:\n"
                "                mod_scale = 4\n"
                "            mod_pad_h = 0\n"
                "            mod_pad_w = 0\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = batch.size()\n"
                "                if h % mod_scale != 0:\n"
                "                    mod_pad_h = mod_scale - h % mod_scale\n"
                "                if w % mod_scale != 0:\n"
                "                    mod_pad_w = mod_scale - w % mod_scale\n"
                "                batch = F.pad(batch, (0, mod_pad_w, 0, mod_pad_h), \"reflect\")\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - mod_pad_h * self.upsampler.scale, 0:w - mod_pad_w * self.upsampler.scale]\n"
                "            if pre_pad != 0:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - pre_pad * self.upsampler.scale, 0:w - pre_pad * self.upsampler.scale]\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output[[2, 1, 0], :, :], (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n"
                "        except Exception:\n"
                "            return [self(image) for image in images]",
            )
        else:
            content = content.replace(
                "        try:\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32) / 255.0\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output, (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n",
                "        try:\n"
                "            if self.upsampler.tile_size > 0:\n"
                "                return [self(image) for image in images]\n"
                "            tensors = []\n"
                "            for image in images:\n"
                "                array = np.array(image).astype(np.float32)\n"
                "                if len(array.shape) != 3 or array.shape[2] != 3:\n"
                "                    return [self(image) for image in images]\n"
                "                max_range = 65535 if np.max(array) > 256 else 255\n"
                "                array = array / max_range\n"
                "                array = array[:, :, [2, 1, 0]]\n"
                "                tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()\n"
                "                tensors.append(tensor)\n"
                "            batch = torch.stack(tensors, dim=0).to(self.upsampler.device)\n"
                "            if self.upsampler.half:\n"
                "                batch = batch.half()\n"
                "            pre_pad = getattr(self.upsampler, \"pre_pad\", 0)\n"
                "            if pre_pad != 0:\n"
                "                batch = F.pad(batch, (0, pre_pad, 0, pre_pad), \"reflect\")\n"
                "            mod_scale = None\n"
                "            if self.upsampler.scale == 2:\n"
                "                mod_scale = 2\n"
                "            elif self.upsampler.scale == 1:\n"
                "                mod_scale = 4\n"
                "            mod_pad_h = 0\n"
                "            mod_pad_w = 0\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = batch.size()\n"
                "                if h % mod_scale != 0:\n"
                "                    mod_pad_h = mod_scale - h % mod_scale\n"
                "                if w % mod_scale != 0:\n"
                "                    mod_pad_w = mod_scale - w % mod_scale\n"
                "                batch = F.pad(batch, (0, mod_pad_w, 0, mod_pad_h), \"reflect\")\n"
                "            with torch.no_grad():\n"
                "                output = self.upsampler.model(batch)\n"
                "            if mod_scale is not None:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - mod_pad_h * self.upsampler.scale, 0:w - mod_pad_w * self.upsampler.scale]\n"
                "            if pre_pad != 0:\n"
                "                _, _, h, w = output.size()\n"
                "                output = output[:, :, 0:h - pre_pad * self.upsampler.scale, 0:w - pre_pad * self.upsampler.scale]\n"
                "            output = output.float().cpu().clamp_(0, 1).numpy()\n"
                "            results = []\n"
                "            for image_output in output:\n"
                "                image_output = np.transpose(image_output[[2, 1, 0], :, :], (1, 2, 0))\n"
                "                image_output = (image_output * 255.0).round().astype(np.uint8)\n"
                "                results.append(Image.fromarray(image_output))\n"
                "            return results\n",
            )
        if content != original:
            image_super_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched RealESRGAN enhancer for batched views")

    multiview_utils = source_dir / "hy3dpaint" / "utils" / "multiview_utils.py"
    if multiview_utils.is_file():
        content = multiview_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):\n"
            "        pils = self.forward_one(\n"
            "            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input\n"
            "        )",
            "    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False, num_inference_steps=None, guidance_scale=None, seed=0):\n"
            "        pils = self.forward_one(\n"
            "            images,\n"
            "            conditions,\n"
            "            prompt=prompt,\n"
            "            custom_view_size=custom_view_size,\n"
            "            resize_input=resize_input,\n"
            "            num_inference_steps=num_inference_steps,\n"
            "            guidance_scale=guidance_scale,\n"
            "            seed=seed,\n"
            "        )",
        )
        content = content.replace(
            "    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):\n"
            "        self.seed_everything(0)",
            "    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False, num_inference_steps=None, guidance_scale=None, seed=0):\n"
            "        self.seed_everything(seed)",
        )
        content = content.replace(
            "        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))",
            "        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(seed))",
        )
        content = content.replace(
            "        mvd_image = self.pipeline(\n"
            "            input_images[0:1],\n"
            "            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],\n"
            "            prompt=prompt,\n"
            "            sync_condition=sync_condition,\n"
            "            guidance_scale=3.0,\n"
            "            **kwargs,\n"
            "        ).images",
            "        steps = num_inference_steps or infer_steps_dict[self.pipeline.scheduler.__class__.__name__]\n"
            "        guidance = 3.0 if guidance_scale is None else guidance_scale\n"
            "        mvd_image = self.pipeline(\n"
            "            input_images[0:1],\n"
            "            num_inference_steps=steps,\n"
            "            prompt=prompt,\n"
            "            sync_condition=sync_condition,\n"
            "            guidance_scale=guidance,\n"
            "            **kwargs,\n"
            "        ).images",
        )
        if content != original:
            multiview_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched texture diffusion options")

    mesh_utils = source_dir / "hy3dpaint" / "DifferentiableRenderer" / "mesh_utils.py"
    if mesh_utils.is_file():
        content = mesh_utils.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "import cv2\nimport bpy\nimport math",
            "import cv2\ntry:\n    import bpy\nexcept ImportError:\n    bpy = None\nimport math",
        )
        if content != original:
            mesh_utils.write_text(content, encoding="utf-8")
            emit("setup", "Patched Hunyuan mesh IO for standalone texture loading")

    mesh_render = source_dir / "hy3dpaint" / "DifferentiableRenderer" / "MeshRender.py"
    if mesh_render.is_file():
        content = mesh_render.read_text(encoding="utf-8")
        original = content
        content = content.replace(
            "try:\n"
            "    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor\n"
            "except:\n"
            '    print("InPaint Function CAN NOT BE Imported!!!")',
            "try:\n"
            "    from .mesh_inpaint_processor import meshVerticeInpaint  # , meshVerticeColor\n"
            "except Exception:\n"
            "    try:\n"
            "        from mesh_inpaint_processor import meshVerticeInpaint\n"
            "    except Exception:\n"
            "        meshVerticeInpaint = None\n"
            '        print("InPaint Function CAN NOT BE Imported!!!")',
        )
        content = content.replace(
            "        if vertex_inpaint:\n"
            "            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()\n"
            "            texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)",
            "        if vertex_inpaint and meshVerticeInpaint is not None:\n"
            "            vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh()\n"
            "            texture_np, mask = meshVerticeInpaint(texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)",
        )
        if content != original:
            mesh_render.write_text(content, encoding="utf-8")
            emit("setup", "Patched Hunyuan texture inpaint import for standalone wheels")

    texture_pipeline = source_dir / "hy3dpaint" / "textureGenPipeline.py"
    if not texture_pipeline.is_file():
        return
    content = texture_pipeline.read_text(encoding="utf-8")
    original = content
    if "_ih3d_emit_texture_timing" not in content:
        content = content.replace(
            "import os\nimport torch",
            "import os\nimport json\nimport time\nimport torch",
            1,
        )
        content = content.replace(
            "diffusers_logging.set_verbosity(50)\n",
            "diffusers_logging.set_verbosity(50)\n"
            "\n"
            "\n"
            "def _ih3d_format_elapsed(seconds):\n"
            "    if seconds < 60:\n"
            "        return f\"{seconds:.1f}s\"\n"
            "    minutes, remaining = divmod(int(seconds), 60)\n"
            "    hours, minutes = divmod(minutes, 60)\n"
            "    if hours:\n"
            "        return f\"{hours}h {minutes:02d}m {remaining:02d}s\"\n"
            "    return f\"{minutes}m {remaining:02d}s\"\n"
            "\n"
            "\n"
            "def _ih3d_emit_texture_timing(config, stage, started):\n"
            "    try:\n"
            "        elapsed = time.perf_counter() - started\n"
            "        payload = {\n"
            "            \"type\": \"stage_timing\",\n"
            "            \"message\": f\"{stage} finished in {_ih3d_format_elapsed(elapsed)}\",\n"
            "            \"stage\": stage,\n"
            "            \"elapsedSeconds\": round(elapsed, 3),\n"
            "        }\n"
            "        event_path = getattr(config, \"ih3d_event_path\", None)\n"
            "        if event_path:\n"
            "            payload[\"path\"] = event_path\n"
            "        print(json.dumps(payload, ensure_ascii=True), flush=True)\n"
            "    except Exception:\n"
            "        pass\n",
            1,
        )
    content = content.replace(
        "from DifferentiableRenderer.mesh_utils import convert_obj_to_glb",
        "from convert_utils import create_glb_with_pbr_materials",
    )
    content = content.replace(
        "        # Process mesh\n"
        "        path = os.path.dirname(mesh_path)\n"
        "        if use_remesh:\n"
        "            processed_mesh_path = os.path.join(path, \"white_mesh_remesh.obj\")\n"
        "            remesh_mesh(mesh_path, processed_mesh_path)\n"
        "        else:\n"
        "            processed_mesh_path = mesh_path\n",
        "        # Process mesh\n"
        "        path = os.path.dirname(mesh_path)\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        if use_remesh:\n"
        "            processed_mesh_path = os.path.join(path, \"white_mesh_remesh.obj\")\n"
        "            remesh_mesh(mesh_path, processed_mesh_path)\n"
        "        else:\n"
        "            processed_mesh_path = mesh_path\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture mesh prep\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        # Load mesh\n"
        "        mesh = trimesh.load(processed_mesh_path)\n"
        "        mesh = mesh_uv_wrap(mesh)\n"
        "        self.render.load_mesh(mesh=mesh)\n",
        "        # Load mesh\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        mesh = trimesh.load(processed_mesh_path)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture mesh load\", _ih3d_stage_started)\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        mesh = mesh_uv_wrap(mesh)\n"
        "        self.render.load_mesh(mesh=mesh)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture UV unwrap\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        ########### View Selection #########\n"
        "        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(\n"
        "            self.config.candidate_camera_elevs,\n"
        "            self.config.candidate_camera_azims,\n"
        "            self.config.candidate_view_weights,\n"
        "            self.config.max_selected_view_num,\n"
        "        )\n"
        "\n"
        "        normal_maps = self.view_processor.render_normal_multiview(\n"
        "            selected_camera_elevs, selected_camera_azims, use_abs_coor=True\n"
        "        )\n"
        "        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)\n",
        "        ########### View Selection #########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(\n"
        "            self.config.candidate_camera_elevs,\n"
        "            self.config.candidate_camera_azims,\n"
        "            self.config.candidate_view_weights,\n"
        "            self.config.max_selected_view_num,\n"
        "        )\n"
        "\n"
        "        normal_maps = self.view_processor.render_normal_multiview(\n"
        "            selected_camera_elevs, selected_camera_azims, use_abs_coor=True\n"
        "        )\n"
        "        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture view render\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        multiviews_pbr = self.models[\"multiview_model\"](\n"
        "            image_style,\n"
        "            normal_maps + position_maps,\n"
        "            prompt=image_caption,\n"
        "            custom_view_size=self.config.resolution,\n"
        "            resize_input=True,\n"
        "        )",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        multiviews_pbr = self.models[\"multiview_model\"](\n"
        "            image_style,\n"
        "            normal_maps + position_maps,\n"
        "            prompt=image_caption,\n"
        "            custom_view_size=self.config.resolution,\n"
        "            resize_input=True,\n"
        "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
        "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
        "            seed=getattr(self.config, \"texture_seed\", 0),\n"
        "        )\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture multiview diffusion\", _ih3d_stage_started)",
    )
    if "Texture multiview diffusion" not in content:
        content = content.replace(
            "        multiviews_pbr = self.models[\"multiview_model\"](\n"
            "            image_style,\n"
            "            normal_maps + position_maps,\n"
            "            prompt=image_caption,\n"
            "            custom_view_size=self.config.resolution,\n"
            "            resize_input=True,\n"
            "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
            "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
            "            seed=getattr(self.config, \"texture_seed\", 0),\n"
            "        )",
            "        _ih3d_stage_started = time.perf_counter()\n"
            "        multiviews_pbr = self.models[\"multiview_model\"](\n"
            "            image_style,\n"
            "            normal_maps + position_maps,\n"
            "            prompt=image_caption,\n"
            "            custom_view_size=self.config.resolution,\n"
            "            resize_input=True,\n"
            "            num_inference_steps=getattr(self.config, \"texture_steps\", None),\n"
            "            guidance_scale=getattr(self.config, \"texture_guidance\", None),\n"
            "            seed=getattr(self.config, \"texture_seed\", 0),\n"
            "        )\n"
            "        _ih3d_emit_texture_timing(self.config, \"Texture multiview diffusion\", _ih3d_stage_started)",
        )
    content = content.replace(
        "        for i in range(len(enhance_images[\"albedo\"])):\n"
        "            enhance_images[\"albedo\"][i] = self.models[\"super_model\"](enhance_images[\"albedo\"][i])\n"
        "            enhance_images[\"mr\"][i] = self.models[\"super_model\"](enhance_images[\"mr\"][i])",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        super_model = self.models[\"super_model\"]\n"
        "        if hasattr(super_model, \"enhance_batch\"):\n"
        "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
        "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
        "        else:\n"
        "            for i in range(len(enhance_images[\"albedo\"])):\n"
        "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
        "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture RealESRGAN enhance\", _ih3d_stage_started)",
    )
    if "Texture RealESRGAN enhance" not in content:
        content = content.replace(
            "        super_model = self.models[\"super_model\"]\n"
            "        if hasattr(super_model, \"enhance_batch\"):\n"
            "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
            "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
            "        else:\n"
            "            for i in range(len(enhance_images[\"albedo\"])):\n"
            "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
            "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])",
            "        _ih3d_stage_started = time.perf_counter()\n"
            "        super_model = self.models[\"super_model\"]\n"
            "        if hasattr(super_model, \"enhance_batch\"):\n"
            "            enhance_images[\"albedo\"] = super_model.enhance_batch(enhance_images[\"albedo\"])\n"
            "            enhance_images[\"mr\"] = super_model.enhance_batch(enhance_images[\"mr\"])\n"
            "        else:\n"
            "            for i in range(len(enhance_images[\"albedo\"])):\n"
            "                enhance_images[\"albedo\"][i] = super_model(enhance_images[\"albedo\"][i])\n"
            "                enhance_images[\"mr\"][i] = super_model(enhance_images[\"mr\"][i])\n"
            "        _ih3d_emit_texture_timing(self.config, \"Texture RealESRGAN enhance\", _ih3d_stage_started)",
        )
    content = content.replace(
        "        ###########  Bake  ##########\n"
        "        for i in range(len(enhance_images)):\n"
        "            enhance_images[\"albedo\"][i] = enhance_images[\"albedo\"][i].resize(\n"
        "                (self.config.render_size, self.config.render_size)\n"
        "            )\n"
        "            enhance_images[\"mr\"][i] = enhance_images[\"mr\"][i].resize((self.config.render_size, self.config.render_size))\n"
        "        texture, mask = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"albedo\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        texture_mr, mask_mr = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"mr\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n",
        "        ###########  Bake  ##########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        for i in range(len(enhance_images)):\n"
        "            enhance_images[\"albedo\"][i] = enhance_images[\"albedo\"][i].resize(\n"
        "                (self.config.render_size, self.config.render_size)\n"
        "            )\n"
        "            enhance_images[\"mr\"][i] = enhance_images[\"mr\"][i].resize((self.config.render_size, self.config.render_size))\n"
        "        texture, mask = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"albedo\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        texture_mr, mask_mr = self.view_processor.bake_from_multiview(\n"
        "            enhance_images[\"mr\"], selected_camera_elevs, selected_camera_azims, selected_view_weights\n"
        "        )\n"
        "        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture bake\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        ##########  inpaint  ###########\n"
        "        texture = self.view_processor.texture_inpaint(texture, mask_np)\n"
        "        self.render.set_texture(texture, force_set=True)\n"
        "        if \"mr\" in enhance_images:\n"
        "            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)\n"
        "            self.render.set_texture_mr(texture_mr)\n",
        "        ##########  inpaint  ###########\n"
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        texture = self.view_processor.texture_inpaint(texture, mask_np)\n"
        "        self.render.set_texture(texture, force_set=True)\n"
        "        if \"mr\" in enhance_images:\n"
        "            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)\n"
        "            self.render.set_texture_mr(texture_mr)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture inpaint\", _ih3d_stage_started)\n",
    )
    content = content.replace(
        "        self.render.save_mesh(output_mesh_path, downsample=True)\n"
        "\n"
        "        if save_glb:\n",
        "        _ih3d_stage_started = time.perf_counter()\n"
        "        self.render.save_mesh(output_mesh_path, downsample=True)\n"
        "        _ih3d_emit_texture_timing(self.config, \"Texture OBJ export\", _ih3d_stage_started)\n"
        "\n"
        "        if save_glb:\n",
    )
    content = content.replace(
        '            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))\n'
        '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
        '            textures = {\n'
        '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
        '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
        '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
        '            }\n'
        '            _ih3d_stage_started = time.perf_counter()\n'
        '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
        '            _ih3d_emit_texture_timing(self.config, "Texture GLB export", _ih3d_stage_started)\n'
        '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
    )
    if "Texture GLB export" not in content:
        content = content.replace(
            '            textures = {\n'
            '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
            '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
            '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
            '            }\n'
            '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
            '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
            '            textures = {\n'
            '                "albedo": output_mesh_path.replace(".obj", ".jpg"),\n'
            '                "metallic": output_mesh_path.replace(".obj", "_metallic.jpg"),\n'
            '                "roughness": output_mesh_path.replace(".obj", "_roughness.jpg"),\n'
            '            }\n'
            '            _ih3d_stage_started = time.perf_counter()\n'
            '            create_glb_with_pbr_materials(output_mesh_path, textures, output_mesh_path.replace(".obj", ".glb"))\n'
            '            _ih3d_emit_texture_timing(self.config, "Texture GLB export", _ih3d_stage_started)\n'
            '            output_glb_path = output_mesh_path.replace(".obj", ".glb")',
        )
    if content != original:
        texture_pipeline.write_text(content, encoding="utf-8")
        emit("setup", "Patched Hunyuan texture export for standalone GLB output")

    write_pbr_convert_utils(source_dir / "hy3dpaint" / "convert_utils.py")


def write_pbr_convert_utils(convert_utils_path: Path) -> None:
    """Materialize hy3dpaint/convert_utils.py so textures embed as standard GLB
    binary bufferView data (correct MIME per image) instead of base64 data-URIs.

    Data-URI textures live in the GLB JSON chunk and are loaded through a path
    governed by the page Content-Security-Policy connect-src; locked-down
    webviews silently drop them, leaving the model untextured. Binary bufferView
    textures load through the GLB loader and render everywhere.
    """
    marker = "create_glb_with_pbr_materials"
    required_texture_line = "        textures.append(texture)\n"
    if convert_utils_path.is_file():
        existing = convert_utils_path.read_text(encoding="utf-8")
        if "binary bufferView" in existing and marker in existing and required_texture_line in existing:
            return
    convert_utils_path.parent.mkdir(parents=True, exist_ok=True)
    convert_utils_path.write_text(PBR_CONVERT_UTILS_SOURCE, encoding="utf-8")
    emit("setup", "Wrote PBR GLB convert_utils with binary texture embedding")


def ensure_wheel_source(paths: dict[str, Path]) -> Path:
    wheel_source_dir = paths["wheel_source_dir"]
    if (wheel_source_dir / "hy3dpaint").is_dir():
        return wheel_source_dir
    emit("setup", "Downloading Windows texture wheels")
    archive = paths["runtime_dir"] / "ComfyUI-Hunyuan3d-2-1-main.zip"
    download_file(WHEEL_SOURCE_URL, archive, "texture wheels")
    safe_extract_zip(archive, wheel_source_dir)
    return wheel_source_dir


def matching_wheel(dist_dir: Path, package_prefix: str, version: tuple[int, int, int]) -> Path | None:
    tag = f"cp{version[0]}{version[1]}"
    candidates = sorted(dist_dir.glob(f"{package_prefix}*{tag}*win_amd64.whl"))
    return candidates[0] if candidates else None


def install_texture_wheels(python_path: Path, paths: dict[str, Path]) -> None:
    marker = paths["env_dir"] / WHEEL_MARKER
    if marker.exists():
        return
    if platform.system() != "Windows":
        build_texture_extensions_from_source(python_path, paths)
        return

    version = python_version(python_path)
    if not supported_python_version(version):
        raise RuntimeError("Texture wheels require Python 3.10, 3.11, or 3.12.")
    assert version is not None

    wheel_source = ensure_wheel_source(paths)
    raster = matching_wheel(
        wheel_source / "hy3dpaint" / "custom_rasterizer" / "dist",
        "custom_rasterizer",
        version,
    )
    painter = matching_wheel(
        wheel_source / "hy3dpaint" / "DifferentiableRenderer" / "dist",
        "mesh_inpaint_processor",
        version,
    )

    missing = []
    if raster is None:
        missing.append("custom_rasterizer")
    if painter is None:
        missing.append("mesh_inpaint_processor")
    if missing:
        tag = f"cp{version[0]}{version[1]}"
        raise RuntimeError(
            f"Missing prebuilt texture wheel(s) for {tag}: {', '.join(missing)}. "
            "Install Python 3.10, 3.11, or 3.12 and run Prepare again."
        )

    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", str(raster)], "Installing texture rasterizer wheel")
    run_checked(pip + ["install", "--upgrade", str(painter)], "Installing texture inpaint wheel")
    marker.write_text("ok\n", encoding="utf-8")


# --- Non-Windows texture extensions -------------------------------------------
# The Windows build installs prebuilt texture wheels. Elsewhere the two native
# texture modules are compiled from the Hunyuan3D source: mesh_inpaint_processor
# (plain C++/pybind11) and custom_rasterizer (a PyTorch CUDA extension). nvcc
# comes from a system CUDA toolkit matching PyTorch's CUDA version when one is
# installed, otherwise from NVIDIA's official redistributable archives
# (checksum-verified) unpacked into the app venv, so no root install is needed.
# Mesh-only generation needs neither module, so a failed build only disables
# texture painting.

TEXTURE_BUILD_ERROR_FILE = ".ih3d-texture-build-error.txt"
CUDA_REDIST_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
# PyTorch CUDA version -> NVIDIA redist release with the same minor version.
CUDA_REDIST_RELEASES = {(12, 4): "12.4.1", (12, 8): "12.8.1"}
CUDA_REDIST_COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl")
# Newest GCC each nvcc release accepts.
NVCC_MAX_GCC = {(12, 4): 13, (12, 6): 13, (12, 8): 14, (12, 9): 14}
# glibc 2.41+ declares these math functions noexcept; CUDA 12.x headers do not.
CUDA_GLIBC_NOEXCEPT_FUNCTIONS = ("rsqrt", "rsqrtf", "sinpi", "sinpif", "cospi", "cospif")


def venv_build_info(python_path: Path) -> dict[str, Any]:
    script = (
        "import json, sysconfig, torch; "
        "print(json.dumps({'cuda': torch.version.cuda, 'platlib': sysconfig.get_paths()['platlib'], "
        "'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX')}))"
    )
    output = subprocess.check_output([str(python_path), "-c", script], text=True, encoding="utf-8", errors="replace")
    return json.loads(output.strip().splitlines()[-1])


def parse_cuda_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"release (\d+)\.(\d+)", text) or re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        output = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return parse_cuda_version(output)


def find_system_cuda_home(wanted: tuple[int, int]) -> Path | None:
    candidates: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]))
    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(Path(nvcc_on_path).resolve().parent.parent)
    candidates += [Path("/opt/cuda"), Path("/usr/local/cuda")]
    for home in candidates:
        version = nvcc_version(home / "bin" / "nvcc")
        # Same major version as PyTorch's CUDA and at least as new (sm_120 needs 12.8+).
        if version and version[0] == wanted[0] and version >= wanted:
            return home
    return None


def patch_cuda_math_headers(home: Path) -> None:
    names = "|".join(CUDA_GLIBC_NOEXCEPT_FUNCTIONS)
    rules = [
        (
            home / "include" / "crt" / "math_functions.h",
            re.compile(rf"^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (?:double|float) +(?:{names})\((?:double|float) x\));$", re.M),
            r"\1 noexcept(true);",
        ),
        (
            home / "include" / "crt" / "math_functions.hpp",
            re.compile(rf"^__func__\(((?:double|float) (?:{names})\((?:const )?(?:double|float) a\))\)$", re.M),
            r"__func__(\1 noexcept(true))",
        ),
    ]
    for header, pattern, replacement in rules:
        if header.is_file():
            content = header.read_text(encoding="utf-8")
            header.write_text(pattern.sub(replacement, content), encoding="utf-8")


def ensure_redist_cuda_home(python_path: Path, info: dict[str, Any], wanted: tuple[int, int]) -> Path:
    release = CUDA_REDIST_RELEASES.get(wanted)
    if release is None:
        raise RuntimeError(
            f"No CUDA {wanted[0]}.{wanted[1]} toolkit was found. Install the CUDA toolkit (nvcc) and run setup again."
        )
    home = python_path.parent.parent / "cuda-toolkit"
    ready = home / ".ready"
    if ready.is_file() and ready.read_text(encoding="utf-8").strip() == release:
        return home
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    manifest_path = home / "redistrib.json"
    download_file(f"{CUDA_REDIST_BASE_URL}/redistrib_{release}.json", manifest_path, f"CUDA {release} manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for component in CUDA_REDIST_COMPONENTS:
        entry = manifest[component]["linux-x86_64"]
        archive = home / Path(entry["relative_path"]).name
        download_file(f"{CUDA_REDIST_BASE_URL}/{entry['relative_path']}", archive, f"CUDA {component}")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"CUDA {component} download checksum did not match.")
        emit("setup", f"Unpacking CUDA {component}")
        with tarfile.open(archive) as bundle:
            # Strip the archive's top-level folder; skip anything unsafe.
            members = []
            for member in bundle.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts or ".." in parts or Path(member.name).is_absolute():
                    continue
                member.name = str(Path(*parts))
                members.append(member)
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(home, members=members, filter="data")
            else:
                bundle.extractall(home, members=members)
        archive.unlink()
    if not (home / "lib64").exists():
        (home / "lib64").symlink_to("lib")
    # cuBLAS/cuSPARSE/cuSOLVER/cuRAND headers come from the NVIDIA pip packages
    # PyTorch already installed.
    nvidia_root = Path(info["platlib"]) / "nvidia"
    if nvidia_root.is_dir():
        for package in sorted(nvidia_root.iterdir()):
            include = package / "include"
            if not include.is_dir():
                continue
            for header in include.iterdir():
                target = home / "include" / header.name
                if header.name != "__init__.py" and not target.exists():
                    target.symlink_to(header)
    patch_cuda_math_headers(home)
    ready.write_text(release + "\n", encoding="utf-8")
    return home


def compiler_major(command: str) -> int | None:
    try:
        output = subprocess.check_output([command, "-dumpversion"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    match = re.match(r"(\d+)", output.strip())
    return int(match.group(1)) if match else None


def nvcc_host_flags(nvcc: tuple[int, int]) -> list[str]:
    max_gcc = NVCC_MAX_GCC.get(nvcc, 14)
    gcc = compiler_major("g++")
    if gcc is not None and gcc <= max_gcc:
        return []
    for version in range(max_gcc, 9, -1):
        versioned = shutil.which(f"g++-{version}")
        if versioned:
            return ["-ccbin", versioned]
    if shutil.which("clang++"):
        # Rolling distros (Arch/Omarchy) ship a GCC newer than nvcc's front end
        # can parse. Compile the host side with clang instead and keep
        # libstdc++ from using the newest compiler builtins and C++26 paths.
        return [
            "-allow-unsupported-compiler",
            "-ccbin",
            "clang++",
            "-D_GLIBCXX_DO_NOT_USE_BUILTIN_TRAITS",
            "-U__cpp_structured_bindings",
            "-D__cpp_structured_bindings=201606L",
        ]
    raise RuntimeError(
        f"nvcc {nvcc[0]}.{nvcc[1]} supports GCC {max_gcc} or older, but g++ is version {gcc}. "
        f"Install g++-{max_gcc} or clang and run setup again."
    )


def cuda_arch_list() -> str:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    caps = sorted({gpu["capability"] for gpu in nvidia_gpus()})
    if not caps:
        caps = [(8, 6), (8, 9), (12, 0)]
    return ";".join(f"{major}.{minor}" for major, minor in caps)


def texture_build_env(python_path: Path, info: dict[str, Any]) -> dict[str, str]:
    cuda = parse_cuda_version(str(info.get("cuda") or ""))
    if cuda is None:
        raise RuntimeError("The app PyTorch build has no CUDA support, so the texture rasterizer cannot be built.")
    home = find_system_cuda_home(cuda)
    if home is None:
        home = ensure_redist_cuda_home(python_path, info, cuda)
        emit("setup", f"Using app-managed CUDA compiler: {home}")
    else:
        emit("setup", f"Using CUDA toolkit: {home}")
    env = venv_subprocess_env(python_path)
    env["CUDA_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list()
    env.setdefault("MAX_JOBS", str(max(1, min(16, os.cpu_count() or 1))))
    flags = nvcc_host_flags(nvcc_version(home / "bin" / "nvcc") or cuda)
    if flags:
        env["NVCC_APPEND_FLAGS"] = " ".join([env.get("NVCC_APPEND_FLAGS", ""), *flags]).strip()
    return env


def build_mesh_inpaint_processor(python_path: Path, info: dict[str, Any], source_dir: Path) -> None:
    cpp = source_dir / "hy3dpaint" / "DifferentiableRenderer" / "mesh_inpaint_processor.cpp"
    if not cpp.is_file():
        raise RuntimeError(f"Hunyuan source is missing {cpp.name}.")
    includes = subprocess.check_output(
        [str(python_path), "-m", "pybind11", "--includes"], text=True, encoding="utf-8", errors="replace"
    ).split()
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("A C++ compiler (g++ or clang++) is required to build the texture inpaint module.")
    target = Path(info["platlib"]) / f"mesh_inpaint_processor{info['ext_suffix']}"
    run_checked(
        [compiler, "-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", *includes, str(cpp), "-o", str(target)],
        "Building texture inpaint module",
    )


def build_texture_extensions_from_source(python_path: Path, paths: dict[str, Path]) -> None:
    marker = paths["env_dir"] / WHEEL_MARKER
    error_file = paths["env_dir"] / TEXTURE_BUILD_ERROR_FILE
    source_dir = paths["hunyuan_source_dir"]
    try:
        info = venv_build_info(python_path)
        build_mesh_inpaint_processor(python_path, info, source_dir)
        env = texture_build_env(python_path, info)
        if not python_can_import(python_path, "ninja"):
            run_checked([str(python_path), "-m", "pip", "install", "ninja"], "Installing ninja build tool")
        run_checked(
            [
                str(python_path), "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                str(source_dir / "hy3dpaint" / "custom_rasterizer"),
            ],
            f"Building texture rasterizer for {env['TORCH_CUDA_ARCH_LIST'].replace(';', ', ')}",
            env=env,
        )
        if not python_can_import(python_path, "custom_rasterizer_kernel"):
            raise RuntimeError("The texture rasterizer built but could not be imported.")
    except Exception as error:
        message = str(error).strip()
        error_file.write_text(message + "\n", encoding="utf-8")
        emit(
            "setup",
            "Texture painting is unavailable: the native texture modules could not be built. "
            "Mesh-only generation still works. Details: " + message.splitlines()[0],
        )
        return
    error_file.unlink(missing_ok=True)
    marker.write_text("source-build\n", encoding="utf-8")


def ensure_pkg_resources(python_path: Path) -> None:
    if python_can_import(python_path, "pkg_resources"):
        return
    pip = [str(python_path), "-m", "pip"]
    run_checked(
        pip + ["install", "--upgrade", "setuptools<81"],
        "Installing setuptools compatibility package",
    )


def ensure_env(paths: dict[str, Path], pytorch_profile: str) -> Path:
    env_dir = paths["env_dir"]
    python_path = venv_python(env_dir)
    marker = env_dir / ENV_MARKER
    env_version = python_version(python_path) if python_path.exists() else None
    # Off Windows, pymeshlab==2022.2.post3 has wheels only up to Python 3.11.
    linux_too_new = platform.system() != "Windows" and env_version is not None and env_version[:2] > (3, 11)
    if python_path.exists() and (not supported_python_version(env_version) or linux_too_new):
        emit("setup", "Recreating app Python environment with a supported Python version")
        shutil.rmtree(env_dir, ignore_errors=True)
        python_path = venv_python(env_dir)

    installed_profile = env_marker_profile(marker)
    if python_path.exists() and marker.exists() and installed_profile != pytorch_profile:
        old_label = PYTORCH_RUNTIME_PROFILES.get(installed_profile or "", {}).get("label", "unknown PyTorch runtime")
        new_label = PYTORCH_RUNTIME_PROFILES[pytorch_profile]["label"]
        emit("setup", f"Recreating app Python environment: {old_label} -> {new_label}")
        shutil.rmtree(env_dir, ignore_errors=True)
        python_path = venv_python(env_dir)

    if platform.system() != "Windows" and (
        (not python_path.exists() and marker.exists())
        or (python_path.exists() and not python_can_import(python_path, "json"))
    ):
        # Made for another OS (e.g. a Windows venv in a shared store) or its
        # base interpreter moved away.
        emit("setup", "Recreating app Python environment: the existing one cannot run here")
        shutil.rmtree(env_dir, ignore_errors=True)

    if not python_path.exists():
        emit("setup", f"Creating app Python environment: {env_dir}")
        if platform.system() == "Windows":
            venv.EnvBuilder(with_pip=True).create(env_dir)
        else:
            # Build from the real interpreter: a venv made through a symlink
            # (e.g. uv's ~/.local/bin/python3.12) points its home at the
            # symlink folder and cannot find the standard library.
            subprocess.check_call([os.path.realpath(sys.executable), "-m", "venv", str(env_dir)])

    if not supported_python_version(python_version(python_path)):
        raise RuntimeError(f"{APP_NAME} requires Python 3.10, 3.11, or 3.12.")

    if marker.exists():
        ensure_pkg_resources(python_path)
        return python_path

    profile = PYTORCH_RUNTIME_PROFILES[pytorch_profile]
    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", *BOOTSTRAP_PACKAGES], "Updating pip")
    run_checked(
        pip + ["install", "--upgrade", *profile["packages"], "--index-url", str(profile["index_url"])],
        f"Installing NVIDIA {profile['label']} runtime",
    )
    run_checked(pip + ["install", "--upgrade", *COMPAT_PACKAGES], "Installing shared runtime packages")
    run_checked(
        pip + ["install", "--upgrade", HUGGINGFACE_HUB_PIN, "safetensors==0.4.4", "tqdm==4.66.5"],
        "Installing model download helpers",
    )
    run_checked(pip + ["install", "--upgrade", "numpy==1.26.4"], "Pinning compatible NumPy")
    ensure_pkg_resources(python_path)
    marker.write_text(f"pytorch={pytorch_profile}\nruntime=base\n", encoding="utf-8")
    return python_path


def ensure_hunyuan_env(python_path: Path, paths: dict[str, Path]) -> None:
    marker = paths["env_dir"] / HUNYUAN_ENV_MARKER
    if marker.exists():
        if platform.system() != "Windows" and not (paths["env_dir"] / WHEEL_MARKER).exists():
            install_texture_wheels(python_path, paths)
        marker.write_text("ok\n", encoding="utf-8")
        return
    pip = [str(python_path), "-m", "pip"]
    run_checked(
        pip + ["install", "--upgrade", *HUNYUAN_RUNTIME_PACKAGES],
        "Installing Hunyuan inference dependencies",
    )
    run_checked(pip + ["install", "--upgrade", "onnxruntime-gpu>=1.20.0"], "Installing GPU ONNX Runtime")
    run_checked(pip + ["install", "--upgrade", "numpy==1.26.4"], "Pinning compatible NumPy")
    install_texture_wheels(python_path, paths)
    ensure_pkg_resources(python_path)
    marker.write_text("ok\n", encoding="utf-8")


def case_insensitive_env_get(env: dict[str, str], key: str) -> str:
    if key.lower() == "path":
        values: list[str] = []
        if key in env:
            values.append(env[key])
        for env_key, value in env.items():
            if env_key.lower() == "path" and env_key != key:
                values.append(value)
        if values:
            seen: set[str] = set()
            parts: list[str] = []
            for value in values:
                for part in value.split(os.pathsep):
                    normalized = part.strip()
                    if not normalized:
                        continue
                    dedupe_key = normalized.lower() if platform.system() == "Windows" else normalized
                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        parts.append(normalized)
            return os.pathsep.join(parts)
    if key in env:
        return env[key]
    for env_key, value in env.items():
        if env_key.lower() == key.lower():
            return value
    return ""


def find_vswhere() -> Path | None:
    for base in (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
    ):
        if not base:
            continue
        candidate = Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if candidate.is_file():
            return candidate
    return None


def candidate_vcvars64_paths() -> list[Path]:
    candidates: list[Path] = []
    vswhere = find_vswhere()
    if vswhere:
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in result.stdout.splitlines():
                install_path = Path(line.strip())
                if install_path:
                    candidates.append(install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        except Exception:
            pass
    for base in (
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramFiles"),
    ):
        if not base:
            continue
        visual_studio = Path(base) / "Microsoft Visual Studio"
        candidates.extend(visual_studio.glob("20*/*/VC/Auxiliary/Build/vcvars64.bat"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen and candidate.is_file():
            seen.add(resolved)
            unique.append(candidate)
    return unique


def windows_msvc_build_env() -> dict[str, str]:
    global MSVC_ENV_CACHE
    if platform.system() != "Windows":
        return {}
    if MSVC_ENV_CACHE is not None:
        return MSVC_ENV_CACHE
    for vcvars in candidate_vcvars64_paths():
        try:
            with TemporaryDirectory() as temp_dir:
                dump_env = Path(temp_dir) / "dump_vs_env.bat"
                dump_env.write_text(
                    f'@echo off\r\ncall "{vcvars}" >nul\r\nset\r\n',
                    encoding="utf-8",
                )
                result = subprocess.run(
                    ["cmd.exe", "/d", "/c", str(dump_env)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode != 0:
                    continue
                env: dict[str, str] = {}
                for line in result.stdout.splitlines():
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    env[key] = value
            compiler_path = shutil.which("cl.exe", path=case_insensitive_env_get(env, "Path"))
            if compiler_path:
                emit("setup", f"Using MSVC compiler: {compiler_path}")
                MSVC_ENV_CACHE = env
                return env
        except Exception as error:
            emit("setup", f"MSVC environment discovery skipped: {error}")
    emit("setup", "MSVC compiler environment was not found; TRELLIS.2 native extensions may fail.")
    MSVC_ENV_CACHE = {}
    return MSVC_ENV_CACHE


def venv_subprocess_env(python_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(windows_msvc_build_env())
    env_dir = python_path.parent.parent
    env["VIRTUAL_ENV"] = str(env_dir)
    path_key = "Path" if platform.system() == "Windows" else "PATH"
    env[path_key] = str(python_path.parent) + os.pathsep + case_insensitive_env_get(env, path_key)
    if platform.system() == "Windows":
        env["PATH"] = env[path_key]
        env["DISTUTILS_USE_SDK"] = "1"
        env["MSSdk"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    return env


def patch_flex_gemm_source(source_dir: Path) -> None:
    sparse_neighbor_map = source_dir / "flex_gemm" / "kernels" / "cuda" / "spconv" / "sparse_neighbor_map.cu"
    changed = False
    if sparse_neighbor_map.is_file():
        try:
            text = sparse_neighbor_map.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = sparse_neighbor_map.read_text(encoding="utf-8-sig")
        patched = text.replace(
            "expanded_keys.data_ptr<T>()",
            "expanded_keys.template data_ptr<T>()",
        ).replace(
            "valid_keys.data_ptr<T>()",
            "valid_keys.template data_ptr<T>()",
        )
        if patched != text:
            sparse_neighbor_map.write_text(patched, encoding="utf-8", newline="\n")
            changed = True

    setup_py = source_dir / "setup.py"
    if setup_py.is_file():
        text = setup_py.read_text(encoding="utf-8")
        patched = text.replace(
            'os.makedirs(os.path.expanduser("~/.flex_gemm"), exist_ok=True)',
            'flex_gemm_cache_dir = os.environ.get("FLEX_GEMM_CACHE_DIR", os.path.expanduser("~/.flex_gemm"))\nos.makedirs(flex_gemm_cache_dir, exist_ok=True)',
        ).replace(
            'dst_cache_path = os.path.expanduser("~/.flex_gemm/autotune_cache.json")',
            'dst_cache_path = os.path.join(flex_gemm_cache_dir, "autotune_cache.json")',
        )
        if patched != text:
            setup_py.write_text(patched, encoding="utf-8", newline="\n")
            changed = True

    if changed:
        emit("setup", "Patched FlexGEMM source for the Windows TRELLIS.2 build")


def patch_o_voxel_source(source_dir: Path) -> None:
    replacements = {
        source_dir / "src" / "convert" / "flexible_dual_grid.cpp": [
            ("1e-6d", "1e-6"),
            ("0.0d", "0.0"),
            ("{int(voxels .size()), 3}", "{static_cast<int64_t>(voxels.size()), 3}"),
            ("{int(dual_vertices.size()), 3}", "{static_cast<int64_t>(dual_vertices.size()), 3}"),
            ("{int(intersected.size()), 3}", "{static_cast<int64_t>(intersected.size()), 3}"),
        ],
        source_dir / "src" / "io" / "filter_neighbor.cpp": [
            ("torch::zeros({N, C}", "torch::zeros({static_cast<int64_t>(N), static_cast<int64_t>(C)}"),
        ],
        source_dir / "src" / "io" / "filter_parent.cpp": [
            ("torch::zeros({N_leaf, C}", "torch::zeros({static_cast<int64_t>(N_leaf), static_cast<int64_t>(C)}"),
        ],
        source_dir / "src" / "io" / "svo.cpp": [
            ("{svo.size()}", "{static_cast<int64_t>(svo.size())}"),
            ("{codes.size()}", "{static_cast<int64_t>(codes.size())}"),
        ],
    }
    changed = False
    for file_path, file_replacements in replacements.items():
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8-sig")
        patched = text
        for old, new in file_replacements:
            patched = patched.replace(old, new)
        if patched != text:
            file_path.write_text(patched, encoding="utf-8", newline="\n")
            changed = True
    if changed:
        emit("setup", "Patched O-Voxel C++ source for the Windows TRELLIS.2 build")


def ensure_trellis_extension(python_path: Path, paths: dict[str, Path], name: str, repo_url: str, branch_or_tag: str | None) -> None:
    destination = paths["trellis_extensions_dir"] / name
    clone_or_update(repo_url, destination, name, branch_or_tag=branch_or_tag, recursive=True)
    if name == "FlexGEMM":
        patch_flex_gemm_source(destination)
    run_checked(
        [str(python_path), "-m", "pip", "install", str(destination), "--no-build-isolation"],
        f"Installing TRELLIS.2 extension {name}",
        env=venv_subprocess_env(python_path),
    )


def ensure_trellis_env(python_path: Path, paths: dict[str, Path]) -> None:
    marker = paths["env_dir"] / TRELLIS_ENV_MARKER
    if marker.exists():
        return
    ensure_trellis_source(paths)
    pip = [str(python_path), "-m", "pip"]
    run_checked(pip + ["install", "--upgrade", *TRELLIS_BASIC_PACKAGES], "Installing TRELLIS.2 Python dependencies")
    run_checked(pip + ["install", "--upgrade", TRELLIS_UTILS3D_PACKAGE], "Installing TRELLIS.2 geometry helpers")
    for name, (repo_url, branch_or_tag) in TRELLIS_EXTENSION_REPOS.items():
        ensure_trellis_extension(python_path, paths, name, repo_url, branch_or_tag)
    o_voxel_dir = paths["trellis_source_dir"] / "o-voxel"
    if not o_voxel_dir.is_dir():
        raise RuntimeError("TRELLIS.2 o-voxel submodule is missing. Run setup again with Git available.")
    patch_o_voxel_source(o_voxel_dir)
    run_checked(
        [str(python_path), "-m", "pip", "install", str(o_voxel_dir), "--no-build-isolation", "--no-deps"],
        "Installing TRELLIS.2 O-Voxel extension",
        env=venv_subprocess_env(python_path),
    )
    marker.write_text("ok\n", encoding="utf-8")


def prepare_managed_environment(app_data_dir: Path, pytorch_profile: str, model_backend: str) -> tuple[Path, dict[str, Path]]:
    paths = app_paths(app_data_dir, pytorch_profile)
    for key in ("runtime_dir", "model_dir", "output_dir", "jobs_dir", "trellis_extensions_dir", "cache_dir", "tmp_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    configure_portable_cache_env(paths)
    python_path = ensure_env(paths, pytorch_profile)
    if model_backend == "trellis2":
        ensure_trellis_source(paths)
        ensure_trellis_env(python_path, paths)
    else:
        source_dir = ensure_hunyuan_source(paths)
        patch_hunyuan_source(source_dir)
        ensure_hunyuan_env(python_path, paths)
    return python_path, paths


def managed_env_vars(app_data_dir: Path, paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    configure_portable_cache_env(paths, env)
    env["I3D_IN_MANAGED_ENV"] = "1"
    env["I3D_HUNYUAN_SOURCE_DIR"] = str(paths["hunyuan_source_dir"])
    env["I3D_TRELLIS_SOURCE_DIR"] = str(paths["trellis_source_dir"])
    env["I3D_APP_DATA_DIR"] = str(app_data_dir)
    env["I3D_ENV_DIR"] = str(paths["env_dir"])
    env["I3D_RUNTIME_DIR"] = str(paths["runtime_dir"])
    env["IH3D_IN_MANAGED_ENV"] = "1"
    env["IH3D_SOURCE_DIR"] = str(paths["hunyuan_source_dir"])
    env.setdefault("ATTN_BACKEND", "sdpa")
    env.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env.pop("TQDM_ASCII", None)
    env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    env["IH3D_APP_DATA_DIR"] = str(app_data_dir)
    return env


def reinvoke_inside_env(python_path: Path, job_path: Path, app_data_dir: Path, paths: dict[str, Path]) -> int:
    command = [str(python_path), str(Path(__file__).resolve()), "--job", str(job_path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=managed_env_vars(app_data_dir, paths),
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def ensure_realesrgan(paths: dict[str, Path]) -> Path:
    ckpt = paths["source_dir"] / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if ckpt.is_file():
        return ckpt
    download_file(REALESRGAN_URL, ckpt, "RealESRGAN texture enhancer")
    return ckpt


def ensure_hunyuan_models(paths: dict[str, Path], include_texture: bool = True) -> None:
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    marker = paths["model_dir"] / HUNYUAN_MODEL_MARKER
    realesrgan = paths["hunyuan_source_dir"] / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if marker.exists() and (not include_texture or realesrgan.is_file()):
        return
    ensure_realesrgan(paths)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    emit("model", "Downloading Hunyuan3D 2.1 model bundle")
    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[
            "README.md",
            "hunyuan3d-dit-v2-1/*",
            "hunyuan3d-vae-v2-1/*",
            "hunyuan3d-paintpbr-v2-1/*",
        ],
        local_files_only=False,
        tqdm_class=hf_snapshot_progress_class("Hunyuan3D model files"),
    )
    if include_texture:
        emit("model", "Downloading DINOv2 texture conditioning model")
        snapshot_download(
            repo_id=DINO_REPO,
            local_files_only=False,
            tqdm_class=hf_snapshot_progress_class("DINOv2 model files"),
        )
    marker.write_text("ok\n", encoding="utf-8")


def ensure_trellis_models(paths: dict[str, Path]) -> None:
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    marker = paths["model_dir"] / TRELLIS_MODEL_MARKER
    if marker.exists():
        return
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    emit("model", "Downloading TRELLIS.2 4B model bundle")
    snapshot_download(
        repo_id=TRELLIS_MODEL_REPO,
        local_files_only=False,
        tqdm_class=hf_snapshot_progress_class("TRELLIS.2 model files"),
    )
    emit("model", "Checking TRELLIS.2 DINOv3 image conditioning model")
    try:
        snapshot_download(
            repo_id=TRELLIS_DINOV3_REPO,
            local_files_only=False,
            tqdm_class=hf_snapshot_progress_class("DINOv3 model files"),
        )
    except Exception as error:
        message = str(error)
        if "gated repo" in message.lower() or "403" in message:
            raise RuntimeError(
                "TRELLIS.2 requires access to the gated Hugging Face model "
                f"{TRELLIS_DINOV3_REPO}. The current Hugging Face token is not authorized. "
                f"Request access at https://huggingface.co/{TRELLIS_DINOV3_REPO}, then run Prepare again "
                "with an authorized HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
            ) from error
        raise
    emit("model", "Checking TRELLIS.2 background removal model")
    try:
        snapshot_download(
            repo_id=TRELLIS_REMBG_REPO,
            local_files_only=False,
            tqdm_class=hf_snapshot_progress_class("RMBG model files"),
        )
    except Exception as error:
        message = str(error)
        if "gated repo" in message.lower() or "403" in message:
            raise RuntimeError(
                "TRELLIS.2 requires access to the gated Hugging Face model "
                f"{TRELLIS_REMBG_REPO}. The current Hugging Face token is not authorized. "
                f"Request access at https://huggingface.co/{TRELLIS_REMBG_REPO}, then run Prepare again "
                "with an authorized HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
            ) from error
        raise
    marker.write_text("ok\n", encoding="utf-8")


def slugify(value: str, fallback: str = "asset") -> str:
    stem = Path(value).stem if value else fallback
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (slug or fallback)[:72]


def unique_target(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output name for {path}")


def as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def as_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def as_float(value: Any, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"


def seed_value(options: dict[str, Any], key: str, mode_key: str) -> int:
    if str(options.get(mode_key, "fixed")) == "randomize":
        return random.randint(0, 999_999_999)
    return as_int(options.get(key), 1, 0, 999_999_999)


def parse_number_list(value: str, default: list[float]) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except Exception:
        return default
    return parsed or default


def texture_view_count(options: dict[str, Any]) -> int:
    azimuths = parse_number_list(str(options.get("cameraAzimuths", "")), [0, 90, 180, 270, 0, 180])
    return max(6, min(9, len(azimuths)))


def texture_seed_value(options: dict[str, Any]) -> int:
    return seed_value(options, "textureSeed", "textureSeedMode") % (2**32)


def apply_texture_options(config: Any, options: dict[str, Any]) -> None:
    config.texture_steps = as_int(options.get("textureSteps"), 15, 1, 100)
    config.texture_guidance = as_float(options.get("textureGuidance"), 3.0, 0, 30)
    config.texture_seed = texture_seed_value(options)
    azimuths = parse_number_list(str(options.get("cameraAzimuths", "")), [0, 90, 180, 270, 0, 180])
    elevations = parse_number_list(str(options.get("cameraElevations", "")), [0, 0, 0, 0, 90, -90])
    weights = parse_number_list(str(options.get("viewWeights", "")), [1, 0.1, 0.5, 0.1, 0.05, 0.05])
    count = min(len(azimuths), len(elevations), len(weights))
    if count >= 1:
        config.candidate_camera_azims = azimuths[:count]
        config.candidate_camera_elevs = elevations[:count]
        config.candidate_view_weights = weights[:count]


def shape_cache_key(options: dict[str, Any]) -> tuple[Any, ...]:
    return (
        MODEL_REPO,
        as_bool(options.get("enableFlashVdm"), True),
        str(options.get("mcAlgo", "mc")) or "mc",
        as_bool(options.get("forceOffload"), False),
        as_bool(options.get("compileModels"), False),
    )


def configure_hunyuan_surface_extractor(
    shape_pipeline: Any,
    surface_extractors: dict[str, Any],
    mc_algo: str,
) -> None:
    if mc_algo not in surface_extractors:
        raise RuntimeError(
            f"Unknown Hunyuan3D surface extractor {mc_algo!r}; "
            f"available extractors: {', '.join(surface_extractors)}"
        )
    shape_pipeline.vae.surface_extractor = surface_extractors[mc_algo]()


def export_hunyuan_latents(
    shape_pipeline: Any,
    latents: Any,
    options: dict[str, Any],
    surface_extractors: dict[str, Any],
) -> list[Any]:
    mc_algo = str(options.get("mcAlgo", "mc")) or "mc"
    configure_hunyuan_surface_extractor(shape_pipeline, surface_extractors, mc_algo)
    export_options = {
        "output_type": "trimesh",
        "box_v": as_float(options.get("boxV"), 1.01, 0.1, 5),
        "octree_resolution": as_int(options.get("octreeResolution"), 384, 64, 2048),
        "mc_level": as_float(options.get("mcLevel"), 0.0, -10, 10),
        "mc_algo": None,
        "num_chunks": as_int(options.get("numChunks"), 8000, 1024, 999999),
        "enable_pbar": False,
    }
    try:
        return shape_pipeline._export(latents, **export_options)
    except RuntimeError as error:
        flash_enabled = getattr(shape_pipeline, "_ih3d_flashvdm_enabled", False)
        if not flash_enabled or "expected a non-empty list of Tensors" not in str(error):
            raise
        emit(
            "setup",
            "Fast VAE mesh decode found no surface candidates. "
            "Retrying with the stable dense decoder...",
        )
        shape_pipeline.vae.enable_flashvdm_decoder(enabled=False)
        shape_pipeline._ih3d_flashvdm_enabled = False
        configure_hunyuan_surface_extractor(shape_pipeline, surface_extractors, mc_algo)
        return shape_pipeline._export(latents, **export_options)


def paint_cache_key(options: dict[str, Any]) -> tuple[Any, ...]:
    return (
        texture_view_count(options),
        as_int(options.get("textureViewSize"), 512, 256, 2048),
        as_int(options.get("textureSize"), 2048, 512, 8192),
    )


def maybe_compile_shape_pipeline(shape_pipeline: Any, torch: Any) -> None:
    if getattr(shape_pipeline, "_ih3d_compiled", False):
        return
    started = time.perf_counter()
    try:
        if hasattr(shape_pipeline, "model"):
            shape_pipeline.model = torch.compile(shape_pipeline.model, mode="reduce-overhead", dynamic=False)
            shape_pipeline._ih3d_compiled = True
            emit_timing("Shape model compile", started)
    except Exception as error:
        emit("setup", f"Torch compile skipped: {error}")


def output_base_name(options: dict[str, Any], input_path: Path, index: int, total: int) -> str:
    requested = str(options.get("outputName", "")).strip()
    if not requested:
        return slugify(input_path.name)
    base = slugify(requested)
    return f"{base}_{index + 1:03d}" if total > 1 else base


def image_has_alpha(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        if image.mode != "RGBA":
            return False
        alpha = image.getchannel("A")
        return alpha.getextrema()[0] < 255


def configure_torch_performance(torch: Any) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        emit("setup", "Enabled TF32 matmul and cuDNN autotuning")
    except Exception as error:
        emit("setup", f"Torch performance tuning skipped: {error}")


def release_cuda_memory(torch: Any | None = None, message: str | None = None) -> None:
    gc.collect()
    try:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    if message:
        emit("setup", message)


def release_cached_pipelines(names: tuple[str, ...], torch: Any | None = None, reason: str | None = None) -> None:
    released = False
    if "shape" in names:
        released = released or PIPELINE_CACHE.get("shape_pipeline") is not None
        PIPELINE_CACHE["shape_pipeline"] = None
        PIPELINE_CACHE["shape_key"] = None
    if "paint" in names:
        released = released or PIPELINE_CACHE.get("paint_pipeline") is not None
        PIPELINE_CACHE["paint_pipeline"] = None
        PIPELINE_CACHE["paint_key"] = None
    if "trellis" in names:
        released = released or PIPELINE_CACHE.get("trellis_pipeline") is not None
        PIPELINE_CACHE["trellis_pipeline"] = None
        PIPELINE_CACHE["trellis_key"] = None
    if "background" in names:
        released = released or PIPELINE_CACHE.get("background_remover") is not None
        PIPELINE_CACHE["background_remover"] = None
    release_cuda_memory(torch, reason if released else None)


def load_condition_image(path: Path, rembg: Any | None = None):
    from PIL import Image

    image = Image.open(path)
    if image.mode == "RGBA" and image_has_alpha(path):
        return image.convert("RGBA"), rembg
    from hy3dshape.rembg import BackgroundRemover

    started = time.perf_counter()
    emit("stage_start", f"Removing background for {path.name}", path=str(path))
    if rembg is None:
        rembg = BackgroundRemover()
    result = rembg(image.convert("RGB"))
    emit_timing("Background removal", started, path=str(path))
    return result, rembg


def find_existing_output(*paths: Path) -> list[str]:
    output = []
    for path in paths:
        if path.is_file():
            output.append(str(path.resolve()))
    return output


def repair_glb_embedded_image_mime_types(path: Path) -> int:
    """Correct embedded image data-URI MIME types without adding dependencies."""
    data = path.read_bytes()
    if len(data) < 12:
        return 0
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        return 0

    chunks: list[tuple[int, bytes]] = []
    json_chunk_index: int | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            return 0
        if chunk_type == 0x4E4F534A:
            json_chunk_index = len(chunks)
        chunks.append((chunk_type, chunk))
        offset += chunk_length
    if offset != len(data) or json_chunk_index is None:
        return 0

    json_chunk = chunks[json_chunk_index][1].rstrip(b" \t\r\n\0")
    document = json.loads(json_chunk.decode("utf-8"))
    changed = 0
    for image in document.get("images") or []:
        uri = image.get("uri")
        if not isinstance(uri, str) or not uri.startswith("data:") or ";base64," not in uri:
            continue
        header, encoded = uri.split(",", 1)
        sample_text = encoded[:32]
        sample_text += "=" * (-len(sample_text) % 4)
        try:
            sample = base64.b64decode(sample_text)
        except Exception:
            continue
        actual_mime = None
        if sample.startswith(b"\xff\xd8\xff"):
            actual_mime = "image/jpeg"
        elif sample.startswith(b"\x89PNG\r\n\x1a\n"):
            actual_mime = "image/png"
        elif sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
            actual_mime = "image/webp"
        if actual_mime is None or header == f"data:{actual_mime};base64":
            continue
        image["uri"] = f"data:{actual_mime};base64,{encoded}"
        changed += 1

    if changed == 0:
        return 0

    updated_json = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    updated_json += b" " * (-len(updated_json) % 4)
    chunks[json_chunk_index] = (0x4E4F534A, updated_json)
    total_length = 12 + sum(8 + len(chunk) for _, chunk in chunks)
    output = bytearray(struct.pack("<4sII", b"glTF", 2, total_length))
    for chunk_type, chunk in chunks:
        output.extend(struct.pack("<II", len(chunk), chunk_type))
        output.extend(chunk)
    temporary = path.with_name(f"{path.name}.mime.tmp")
    temporary.write_bytes(output)
    temporary.replace(path)
    return changed


def _read_glb_accessor(gltf: Any, blob: bytes, accessor_index: int):
    import numpy as np

    component_dtypes = {
        5120: np.dtype("<i1"),
        5121: np.dtype("<u1"),
        5122: np.dtype("<i2"),
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    accessor = gltf.accessors[accessor_index]
    if accessor.bufferView is None or accessor.sparse is not None:
        raise RuntimeError("Sparse or bufferless GLB accessors are not supported for normal smoothing.")
    view = gltf.bufferViews[accessor.bufferView]
    dtype = component_dtypes.get(accessor.componentType)
    components = component_counts.get(accessor.type)
    if dtype is None or components is None:
        raise RuntimeError(f"Unsupported GLB accessor format: {accessor.componentType}/{accessor.type}")
    packed_stride = dtype.itemsize * components
    stride = int(view.byteStride or packed_stride)
    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    if offset < 0 or offset + max(0, accessor.count - 1) * stride + packed_stride > len(blob):
        raise RuntimeError("GLB accessor points outside the binary buffer.")
    return np.ndarray(
        shape=(accessor.count, components),
        dtype=dtype,
        buffer=blob,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()


def smooth_glb_normals(path: Path) -> int:
    """Write area-weighted vertex normals while preserving UV seams and textures."""
    import numpy as np
    from pygltflib import Accessor, BufferView, GLTF2

    gltf = GLTF2().load(str(path))
    if not gltf.buffers:
        return 0
    gltf.bufferViews = gltf.bufferViews or []
    gltf.accessors = gltf.accessors or []
    blob = bytearray(gltf.binary_blob())
    primitives_updated = 0

    for mesh in gltf.meshes or []:
        entries: list[tuple[Any, Any, Any]] = []
        all_positions = []
        for primitive in mesh.primitives or []:
            if primitive.mode not in (None, 4) or primitive.attributes.POSITION is None:
                continue
            positions = _read_glb_accessor(
                gltf,
                bytes(blob),
                primitive.attributes.POSITION,
            ).astype(np.float64, copy=False)
            if positions.shape[1] != 3:
                continue
            if primitive.indices is None:
                indices = np.arange(len(positions), dtype=np.int64)
            else:
                indices = _read_glb_accessor(gltf, bytes(blob), primitive.indices).reshape(-1).astype(np.int64)
            usable = len(indices) - (len(indices) % 3)
            if usable < 3:
                continue
            triangles = indices[:usable].reshape(-1, 3)
            valid = np.all((triangles >= 0) & (triangles < len(positions)), axis=1)
            triangles = triangles[valid]
            if len(triangles) == 0:
                continue
            entries.append((primitive, positions, triangles))
            all_positions.append(positions)

        if not entries:
            continue
        combined = np.concatenate(all_positions, axis=0)
        span = float(np.ptp(combined, axis=0).max(initial=0.0))
        tolerance = max(span * 1e-6, 1e-9)
        accumulated: dict[tuple[int, int, int], Any] = {}
        entry_keys: list[Any] = []

        for _, positions, triangles in entries:
            keys = np.rint(positions / tolerance).astype(np.int64)
            entry_keys.append(keys)
            for a, b, c in triangles:
                normal = np.cross(positions[b] - positions[a], positions[c] - positions[a])
                if float(np.dot(normal, normal)) <= 1e-24:
                    continue
                for vertex in (a, b, c):
                    key = tuple(int(value) for value in keys[vertex])
                    current = accumulated.get(key)
                    if current is None:
                        accumulated[key] = normal.copy()
                    else:
                        current += normal

        for (primitive, positions, _), keys in zip(entries, entry_keys):
            normals = np.empty((len(positions), 3), dtype=np.float32)
            for vertex, key_values in enumerate(keys):
                key = tuple(int(value) for value in key_values)
                normal = accumulated.get(key)
                length = float(np.linalg.norm(normal)) if normal is not None else 0.0
                normals[vertex] = normal / length if length > 1e-12 else (0.0, 1.0, 0.0)

            while len(blob) % 4:
                blob.append(0)
            byte_offset = len(blob)
            normal_bytes = normals.astype("<f4", copy=False).tobytes()
            blob.extend(normal_bytes)
            gltf.bufferViews.append(
                BufferView(
                    buffer=0,
                    byteOffset=byte_offset,
                    byteLength=len(normal_bytes),
                    target=34962,
                )
            )
            gltf.accessors.append(
                Accessor(
                    bufferView=len(gltf.bufferViews) - 1,
                    byteOffset=0,
                    componentType=5126,
                    count=len(normals),
                    type="VEC3",
                    min=normals.min(axis=0).tolist(),
                    max=normals.max(axis=0).tolist(),
                )
            )
            primitive.attributes.NORMAL = len(gltf.accessors) - 1
            primitives_updated += 1

    if primitives_updated == 0:
        return 0
    while len(blob) % 4:
        blob.append(0)
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    temporary = path.with_name(f"{path.stem}.normals.tmp.glb")
    gltf.save(str(temporary))
    temporary.replace(path)
    return primitives_updated


def maybe_smooth_exported_glb(path: Path, options: dict[str, Any], image_path: Path | None = None) -> int:
    if not as_bool(options.get("smoothNormals"), True) or path.suffix.lower() != ".glb":
        return 0
    primitives = smooth_glb_normals(path)
    if primitives:
        emit(
            "setup",
            f"Smoothed normals for {primitives} GLB mesh primitive"
            f"{'s' if primitives != 1 else ''}.",
            path=str(image_path) if image_path else None,
        )
    return primitives


def cleanup_job_scratch(jobs_dir: Path, keep: Path | None = None) -> None:
    if not jobs_dir.is_dir():
        return
    keep_resolved = keep.resolve() if keep else None
    for entry in jobs_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "history":
            continue
        try:
            if keep_resolved and entry.resolve() == keep_resolved:
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except Exception:
            continue


PIPELINE_CACHE: dict[str, Any] = {
    "shape_key": None,
    "shape_pipeline": None,
    "paint_key": None,
    "paint_pipeline": None,
    "trellis_key": None,
    "trellis_pipeline": None,
    "background_remover": None,
    "torch_configured": False,
}


def voxel_sidecar_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".ih3dvox.json")


def write_voxel_sidecar(sidecar: dict[str, Any], mesh_path: Path) -> Path:
    sidecar_path = voxel_sidecar_path(mesh_path)
    payload = {
        **sidecar,
        "modelFile": mesh_path.name,
    }
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return sidecar_path


def voxelize_mesh(mesh: Any, resolution: int) -> tuple[Any, dict[str, Any] | None, int]:
    """Convert a smooth mesh into a blocky voxel-style mesh.

    The mesh is sampled into a voxel grid whose longest side is `resolution`
    voxels, then rebuilt as axis-aligned cubes (Minecraft-style)."""
    extents = getattr(mesh, "extents", None)
    if extents is None or max(extents) <= 0:
        raise RuntimeError("Mesh has no volume to voxelize.")
    pitch = float(max(extents)) / float(max(8, resolution))
    voxel_grid = mesh.voxelized(pitch)
    try:
        voxel_grid = voxel_grid.fill()
    except Exception:
        # Filling the interior is optional; a hollow shell still looks correct.
        pass
    sparse_indices = voxel_grid.sparse_indices
    voxel_count = int(len(sparse_indices))
    sidecar = None
    if voxel_count <= VOXEL_DSS_SIDECAR_MAX_CELLS:
        dims = [int(value) for value in voxel_grid.shape]
        sidecar = {
            "format": "ih3d.voxel-dss.v1",
            "dims": dims,
            "cells": [int(value) for row in sparse_indices for value in row],
            "voxelCount": voxel_count,
            "resolution": int(resolution),
            "pitch": float(pitch),
        }
    voxel_mesh = voxel_grid.as_boxes()
    # Preserve the original placement so downstream texturing lines up.
    voxel_mesh.apply_translation(mesh.bounds.mean(axis=0) - voxel_mesh.bounds.mean(axis=0))
    return voxel_mesh, sidecar, voxel_count


def _face_count_from_trimesh(mesh_path: Path) -> int | None:
    try:
        import trimesh

        loaded = trimesh.load(str(mesh_path), force="mesh")
        faces = getattr(loaded, "faces", None)
        if faces is not None:
            return int(len(faces))
        geometry = getattr(loaded, "geometry", None)
        if geometry:
            return int(sum(len(getattr(mesh, "faces", [])) for mesh in geometry.values()))
    except Exception:
        return None
    return None


def _simplify_mesh_with_pymeshlab(mesh_path: Path, target_faces: int) -> tuple[int | None, int | None]:
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(mesh_path))
    before = int(mesh_set.current_mesh().face_number())
    if before <= target_faces:
        return before, before

    mesh_set.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(target_faces),
        preservenormal=True,
        preserveboundary=True,
        preservetopology=False,
        optimalplacement=True,
        qualitythr=1.0,
    )
    after = int(mesh_set.current_mesh().face_number())
    with TemporaryDirectory() as temp_dir:
        simplified_obj = Path(temp_dir) / "simplified.obj"
        mesh_set.save_current_mesh(str(simplified_obj), save_textures=False)
        import trimesh

        simplified_mesh = trimesh.load(str(simplified_obj), force="mesh")
        simplified_mesh.export(str(mesh_path))
    return before, after


def _simplify_mesh_with_trimesh(mesh_path: Path, target_faces: int) -> tuple[int | None, int | None]:
    import trimesh

    loaded = trimesh.load(str(mesh_path), force="mesh")
    faces = getattr(loaded, "faces", None)
    before = int(len(faces)) if faces is not None else None
    if before is not None and before <= target_faces:
        return before, before

    simplified = None
    for method_name in ("simplify_quadric_decimation", "simplify_quadratic_decimation"):
        method = getattr(loaded, method_name, None)
        if method is None:
            continue
        try:
            simplified = method(face_count=int(target_faces))
        except TypeError:
            simplified = method(int(target_faces))
        break
    if simplified is None:
        raise RuntimeError("trimesh simplification support is not available")

    simplified_path = mesh_path.with_name(f"{mesh_path.stem}.simplified{mesh_path.suffix}")
    simplified.export(str(simplified_path))
    simplified_path.replace(mesh_path)
    simplified_faces = getattr(simplified, "faces", None)
    after = int(len(simplified_faces)) if simplified_faces is not None else _face_count_from_trimesh(mesh_path)
    return before, after


def simplify_mesh_file(mesh_path: Path, target_faces: int) -> tuple[int | None, int | None]:
    errors: list[str] = []
    for label, simplifier in (
        ("PyMeshLab", _simplify_mesh_with_pymeshlab),
        ("trimesh", _simplify_mesh_with_trimesh),
    ):
        try:
            return simplifier(mesh_path, target_faces)
        except Exception as error:
            errors.append(f"{label}: {error}")
    raise RuntimeError("; ".join(errors))


def reduce_exported_mesh_faces(mesh_path: Path, target_faces: int, image_path: Path) -> None:
    if not mesh_path.is_file():
        return
    before_hint = _face_count_from_trimesh(mesh_path)
    if before_hint is not None and before_hint <= target_faces:
        emit(
            "setup",
            f"Mesh already fits the {target_faces:,} face target ({before_hint:,} faces).",
            path=str(image_path),
        )
        return
    try:
        before, after = simplify_mesh_file(mesh_path, target_faces)
    except Exception as error:
        emit(
            "setup",
            f"Mesh face reduction skipped: {error}",
            path=str(image_path),
        )
        return

    if before is None:
        before = before_hint
    if after is None:
        after = _face_count_from_trimesh(mesh_path)
    if before is not None and after is not None:
        emit(
            "setup",
            f"Reduced mesh faces from {before:,} to {after:,} (target {target_faces:,}).",
            path=str(image_path),
        )
    elif after is not None:
        emit("setup", f"Reduced mesh to {after:,} faces (target {target_faces:,}).", path=str(image_path))


def patch_transformers_weight_constants_for_diffusers() -> None:
    try:
        import transformers.utils as transformers_utils
    except Exception as error:
        emit("setup", f"Transformers compatibility shim skipped: {error}")
        return

    constants = {
        "WEIGHTS_NAME": "pytorch_model.bin",
        "SAFE_WEIGHTS_NAME": "model.safetensors",
        "TF2_WEIGHTS_NAME": "tf_model.h5",
        "TF_WEIGHTS_NAME": "model.ckpt",
        "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
        "ONNX_WEIGHTS_NAME": "model.onnx",
        "ONNX_EXTERNAL_WEIGHTS_NAME": "weights.pb",
    }
    missing = []
    for name, value in constants.items():
        if not hasattr(transformers_utils, name):
            setattr(transformers_utils, name, value)
            missing.append(name)
    if missing:
        emit("setup", f"Patched Transformers 5 constants for diffusers: {', '.join(missing)}")


def get_hunyuan_paint_pipeline(
    source_dir: Path,
    options: dict[str, Any],
    Hunyuan3DPaintConfig: Any,
    Hunyuan3DPaintPipeline: Any,
) -> Any:
    key = paint_cache_key(options)
    paint_pipeline = PIPELINE_CACHE.get("paint_pipeline") if PIPELINE_CACHE.get("paint_key") == key else None
    if paint_pipeline is None:
        model_started = time.perf_counter()
        emit("model", "Loading Hunyuan3D PBR texture model on CUDA")
        max_num_view = texture_view_count(options)
        resolution = as_int(options.get("textureViewSize"), 512, 256, 2048)
        config = Hunyuan3DPaintConfig(max_num_view, resolution)
        config.realesrgan_ckpt_path = str(source_dir / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth")
        config.multiview_cfg_path = str(source_dir / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.custom_pipeline = str(source_dir / "hy3dpaint" / "hunyuanpaintpbr")
        if hasattr(config, "texture_size"):
            config.texture_size = as_int(options.get("textureSize"), getattr(config, "texture_size"), 512, 8192)
        apply_texture_options(config, options)
        paint_pipeline = Hunyuan3DPaintPipeline(config)
        PIPELINE_CACHE["paint_pipeline"] = paint_pipeline
        PIPELINE_CACHE["paint_key"] = key
        emit_timing("Texture model load", model_started)
    else:
        apply_texture_options(paint_pipeline.config, options)
        emit("model", "Reusing loaded Hunyuan3D PBR texture model")
    shader_type = "vertex" if as_bool(options.get("smoothNormals"), True) else "face"
    if hasattr(paint_pipeline, "render") and hasattr(paint_pipeline.render, "shader_type"):
        previous_shader_type = paint_pipeline.render.shader_type
        paint_pipeline.render.shader_type = shader_type
        if previous_shader_type != shader_type:
            emit(
                "setup",
                "Texture conditioning uses "
                f"{'smooth vertex' if shader_type == 'vertex' else 'flat face'} normals.",
            )
    return paint_pipeline


def run_hunyuan_generation(job: dict[str, Any], paths: dict[str, Path]) -> None:
    source_dir = paths["source_dir"]
    launch_cwd = Path.cwd()
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(source_dir / "hy3dshape"))
    sys.path.insert(0, str(source_dir / "hy3dpaint"))
    os.chdir(source_dir)

    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception as error:
        emit("setup", f"torchvision compatibility fix skipped: {error}")

    patch_transformers_weight_constants_for_diffusers()

    import torch
    from hy3dshape.models.autoencoders import SurfaceExtractors
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    if not PIPELINE_CACHE.get("torch_configured"):
        configure_torch_performance(torch)
        PIPELINE_CACHE["torch_configured"] = True
    release_cached_pipelines(("trellis",), torch, "Released TRELLIS.2 model before Hunyuan3D generation")

    options = job.get("options", {})
    mode = str(options.get("mode", "meshTexture"))
    if mode != "meshOnly" and platform.system() != "Windows":
        import importlib.util

        if importlib.util.find_spec("custom_rasterizer_kernel") is None:
            error_file = paths["env_dir"] / TEXTURE_BUILD_ERROR_FILE
            detail = error_file.read_text(encoding="utf-8").strip() if error_file.is_file() else ""
            raise RuntimeError(
                "Texture painting needs the custom_rasterizer CUDA extension, which is not built in this "
                "runtime. Use Mesh only, or run setup again to retry the build."
                + (f"\n\nLast build error:\n{detail}" if detail else "")
            )
    overwrite = as_bool(options.get("overwrite"), False)
    requested_output = Path(str(options.get("outputDir", "")).strip() or paths["output_dir"])
    if not requested_output.is_absolute():
        requested_output = (launch_cwd / requested_output).resolve()
    requested_output.mkdir(parents=True, exist_ok=True)
    cleanup_job_scratch(paths["jobs_dir"])
    run_dir = paths["jobs_dir"] / job.get("job_id", f"job-{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in job.get("paths", [])]
    total = len(input_paths)
    failed_count = 0
    batch_started = time.perf_counter()
    begin_job_manifest(job, paths, "hunyuan")

    try:
        shape_pipeline = None
        if mode != "textureOnly":
            key = shape_cache_key(options)
            shape_pipeline = PIPELINE_CACHE.get("shape_pipeline") if PIPELINE_CACHE.get("shape_key") == key else None
            if shape_pipeline is None:
                model_started = time.perf_counter()
                emit("model", "Loading Hunyuan3D shape model on CUDA")
                shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_REPO)
                PIPELINE_CACHE["shape_pipeline"] = shape_pipeline
                PIPELINE_CACHE["shape_key"] = key
                emit_timing("Shape model load", model_started)
            else:
                emit("model", "Reusing loaded Hunyuan3D shape model")
            octree_resolution = as_int(options.get("octreeResolution"), 384, 64, 2048)
            flash_requested = as_bool(options.get("enableFlashVdm"), False)
            flash_enabled = flash_requested and octree_resolution > 256
            if flash_enabled:
                flash_started = time.perf_counter()
                if getattr(shape_pipeline, "_ih3d_flashvdm_enabled", False):
                    pass
                elif hasattr(shape_pipeline, "enable_flashvdm"):
                    try:
                        shape_pipeline.enable_flashvdm(
                            enabled=True,
                            adaptive_kv_selection=False,
                            mc_algo=str(options.get("mcAlgo", "mc")) or "mc",
                            replace_vae=False,
                        )
                        shape_pipeline._ih3d_flashvdm_enabled = True
                        emit_timing("FlashVDM setup", flash_started)
                    except Exception as error:
                        emit("setup", f"FlashVDM setup skipped: {error}")
                else:
                    emit("setup", "FlashVDM setup skipped: upstream pipeline does not expose enable_flashvdm")
            elif getattr(shape_pipeline, "_ih3d_flashvdm_enabled", False):
                shape_pipeline.vae.enable_flashvdm_decoder(enabled=False)
                shape_pipeline._ih3d_flashvdm_enabled = False
            if flash_requested and not flash_enabled:
                emit(
                    "setup",
                    "Using the stable VAE mesh decoder at 256 octree resolution.",
                )
            if as_bool(options.get("forceOffload"), False) and hasattr(shape_pipeline, "enable_model_cpu_offload"):
                if not getattr(shape_pipeline, "_ih3d_cpu_offload_enabled", False):
                    shape_pipeline.enable_model_cpu_offload()
                    shape_pipeline._ih3d_cpu_offload_enabled = True
            if as_bool(options.get("compileModels"), False):
                maybe_compile_shape_pipeline(shape_pipeline, torch)

        paint_pipeline = None
        if mode == "textureOnly":
            paint_pipeline = get_hunyuan_paint_pipeline(
                source_dir,
                options,
                Hunyuan3DPaintConfig,
                Hunyuan3DPaintPipeline,
            )

        background_remover = PIPELINE_CACHE.get("background_remover")
        for index, image_path in enumerate(input_paths):
            image_started = time.perf_counter()
            emit("image_start", f"Processing {image_path.name}", path=str(image_path))
            try:
                if not image_path.is_file():
                    raise RuntimeError(f"Input image was not found: {image_path}")

                base = output_base_name(options, image_path, index, total)
                final_mesh_path = unique_target(requested_output / f"{base}.glb", overwrite)
                intermediate_mesh = run_dir / f"{base}_mesh.glb"
                outputs: list[str] = []
                voxel_sidecar: dict[str, Any] | None = None

                if mode != "textureOnly":
                    assert shape_pipeline is not None
                    image, background_remover = load_condition_image(image_path, background_remover)
                    PIPELINE_CACHE["background_remover"] = background_remover
                    seed = seed_value(options, "meshSeed", "meshSeedMode")
                    generator = torch.Generator(device="cuda").manual_seed(seed % (2**32))
                    stage_started = time.perf_counter()
                    emit("stage_start", f"Generating mesh for {image_path.name}", path=str(image_path))
                    latents = shape_pipeline(
                        image=image,
                        num_inference_steps=as_int(options.get("meshSteps"), 50, 1, 200),
                        guidance_scale=as_float(options.get("meshGuidance"), 5.0, 0, 30),
                        generator=generator,
                        output_type="latent",
                        mc_algo=None,
                        enable_pbar=False,
                    )
                    meshes = export_hunyuan_latents(
                        shape_pipeline,
                        latents,
                        options,
                        SurfaceExtractors,
                    )
                    del latents
                    emit_timing("Mesh generation", stage_started, path=str(image_path))
                    mesh = meshes[0]
                    del meshes
                    if as_bool(options.get("voxelize"), False):
                        voxel_started = time.perf_counter()
                        emit("stage_start", f"Voxelizing mesh for {image_path.name}", path=str(image_path))
                        mesh, voxel_sidecar, voxel_count = voxelize_mesh(
                            mesh,
                            as_int(options.get("voxelResolution"), 64, 8, 512),
                        )
                        if voxel_sidecar is None:
                            emit(
                                "setup",
                                f"DSS preview skipped for {image_path.name}: {voxel_count:,} voxels is too large.",
                                path=str(image_path),
                            )
                        emit_timing("Mesh voxelize", voxel_started, path=str(image_path))
                    export_started = time.perf_counter()
                    mesh.export(intermediate_mesh)
                    del mesh
                    generator = None
                    image = None
                    emit_timing("Mesh export", export_started, path=str(image_path))
                    if as_bool(options.get("reduceFaces"), True) and not as_bool(options.get("voxelize"), False):
                        simplify_started = time.perf_counter()
                        target_faces = as_int(options.get("maxFaces"), 5_000, 1_000, 5_000_000)
                        emit(
                            "stage_start",
                            f"Reducing mesh to {target_faces:,} faces for {image_path.name}",
                            path=str(image_path),
                        )
                        reduce_exported_mesh_faces(intermediate_mesh, target_faces, image_path)
                        emit_timing("Mesh simplification", simplify_started, path=str(image_path))
                    emit("stage_done", f"Mesh generated for {image_path.name}", path=str(image_path))
                    if mode == "meshTexture":
                        emit(
                            "mesh_ready",
                            f"Mesh ready for {image_path.name}. Generating PBR texture...",
                            path=str(image_path),
                            output=str(intermediate_mesh.resolve()),
                        )

                if mode == "meshOnly":
                    shutil.move(str(intermediate_mesh), str(final_mesh_path))
                    maybe_smooth_exported_glb(final_mesh_path, options, image_path)
                    outputs.append(str(final_mesh_path.resolve()))
                else:
                    if mode == "meshTexture":
                        shape_pipeline = None
                        release_cached_pipelines(
                            ("shape",),
                            torch,
                            "Released Hunyuan3D shape model before PBR texturing",
                        )
                    if paint_pipeline is None:
                        paint_pipeline = get_hunyuan_paint_pipeline(
                            source_dir,
                            options,
                            Hunyuan3DPaintConfig,
                            Hunyuan3DPaintPipeline,
                        )
                    mesh_source = (
                        Path(str(options.get("meshInputPath", "")).strip())
                        if mode == "textureOnly"
                        else intermediate_mesh
                    )
                    if not mesh_source.is_file():
                        raise RuntimeError(f"Source mesh was not found: {mesh_source}")
                    output_obj = final_mesh_path.with_suffix(".obj")
                    stage_started = time.perf_counter()
                    emit("stage_start", f"Generating PBR texture for {image_path.name}", path=str(image_path))
                    paint_pipeline.config.ih3d_event_path = str(image_path)
                    result = paint_pipeline(
                        mesh_path=str(mesh_source),
                        image_path=str(image_path),
                        output_mesh_path=str(output_obj),
                        use_remesh=False,
                        save_glb=True,
                    )
                    emit_timing("Texture generation", stage_started, path=str(image_path))
                    result_path = Path(str(result)) if result else output_obj
                    glb_candidate = output_obj.with_suffix(".glb")
                    if glb_candidate.exists() and glb_candidate != final_mesh_path:
                        if final_mesh_path.exists() and overwrite:
                            final_mesh_path.unlink()
                        shutil.move(str(glb_candidate), str(final_mesh_path))
                    elif result_path.suffix.lower() == ".glb" and result_path.exists() and result_path != final_mesh_path:
                        if final_mesh_path.exists() and overwrite:
                            final_mesh_path.unlink()
                        shutil.move(str(result_path), str(final_mesh_path))
                    if final_mesh_path.is_file():
                        maybe_smooth_exported_glb(final_mesh_path, options, image_path)
                        repaired_images = repair_glb_embedded_image_mime_types(final_mesh_path)
                        if repaired_images:
                            emit(
                                "setup",
                                f"Corrected {repaired_images} embedded texture MIME type"
                                f"{'s' if repaired_images != 1 else ''} for GLB preview compatibility.",
                                path=str(image_path),
                            )
                        emit(
                            "texture_ready",
                            f"Textured GLB ready for {image_path.name}.",
                            path=str(image_path),
                            output=str(final_mesh_path.resolve()),
                        )
                    outputs.extend(find_existing_output(final_mesh_path, output_obj, output_obj.with_suffix(".jpg")))

                if voxel_sidecar and final_mesh_path.exists():
                    write_voxel_sidecar(voxel_sidecar, final_mesh_path)

                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_done",
                    f"Saved output in {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    outputs=outputs,
                    elapsedSeconds=round(elapsed_seconds, 3),
                )
            except Exception as error:
                failed_count += 1
                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_error",
                    f"{error} after {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    elapsedSeconds=round(elapsed_seconds, 3),
                    traceback=traceback.format_exc(),
                )

        batch_elapsed_seconds = time.perf_counter() - batch_started
        batch_elapsed = format_elapsed(batch_elapsed_seconds)
        if failed_count:
            emit(
                "done",
                f"Batch finished with errors in {batch_elapsed}.",
                failedCount=failed_count,
                totalCount=total,
                elapsedSeconds=round(batch_elapsed_seconds, 3),
            )
        else:
            emit(
                "done",
                f"Batch finished in {batch_elapsed}.",
                failedCount=0,
                totalCount=total,
                elapsedSeconds=round(batch_elapsed_seconds, 3),
            )
    finally:
        if not as_bool(options.get("keepModelsWarm"), True):
            release_cached_pipelines(
                ("shape", "paint", "background"),
                torch,
                "Released Hunyuan3D models after batch",
            )
        shutil.rmtree(run_dir, ignore_errors=True)
        os.chdir(launch_cwd)


def trellis_has_alpha(image: Any) -> bool:
    if getattr(image, "mode", "") != "RGBA":
        return False
    import numpy as np

    return bool(np.asarray(image)[..., 3].min() < 250)


def trellis_autocrop_subject(
    pipeline: Any,
    image: Any,
    probe_size: int = 1024,
    output_size: int = 2048,
    margin: float = 0.06,
) -> tuple[Any, dict[str, Any] | None]:
    """Crop the subject before TRELLIS downsizes the complete source image."""
    from PIL import Image
    import numpy as np

    if trellis_has_alpha(image):
        alpha = np.asarray(image)[..., 3]
        ys, xs = np.nonzero(alpha > 8)
        if len(ys) < 16:
            return image, None
        full_width, full_height = image.size
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = int(round(margin * max(x1 - x0, y1 - y0)))
        box = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(full_width, x1 + pad),
            min(full_height, y1 + pad),
        )
        cropped = image.crop(box)
        return cropped, {
            "method": "alpha",
            "source": [full_width, full_height],
            "crop": list(box),
            "output": list(cropped.size),
        }

    rgb = image.convert("RGB")
    full_width, full_height = rgb.size
    probe = rgb
    if max(probe.size) > probe_size:
        scale = probe_size / max(probe.size)
        probe = probe.resize(
            (
                max(1, round(full_width * scale)),
                max(1, round(full_height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    probe_mask = np.asarray(pipeline.rembg_model(probe))[..., 3]
    ys, xs = np.nonzero(probe_mask > 0.5 * 255)
    if len(ys) < 16:
        return image, None

    scale_x, scale_y = full_width / probe.width, full_height / probe.height
    x0, x1 = xs.min() * scale_x, (xs.max() + 1) * scale_x
    y0, y1 = ys.min() * scale_y, (ys.max() + 1) * scale_y
    pad = margin * max(x1 - x0, y1 - y0)
    box = (
        int(max(0, x0 - pad)),
        int(max(0, y0 - pad)),
        int(min(full_width, x1 + pad)),
        int(min(full_height, y1 + pad)),
    )
    cropped = rgb.crop(box)
    if max(cropped.size) > output_size:
        scale = output_size / max(cropped.size)
        cropped = cropped.resize(
            (
                max(1, round(cropped.width * scale)),
                max(1, round(cropped.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    isolated = pipeline.rembg_model(cropped)
    return isolated, {
        "method": "rmbg",
        "source": [full_width, full_height],
        "crop": list(box),
        "output": list(isolated.size),
        "subjectScale": round(max(full_width, full_height) / max(1, max(box[2] - box[0], box[3] - box[1])), 3),
    }


def trellis_uv_coverage_mask(model_path: Path, width: int, height: int) -> Any:
    import cv2
    import numpy as np
    import trimesh

    loaded = trimesh.load(str(model_path), process=False, force="scene")
    geometries = list(getattr(loaded, "geometry", {}).values())
    if not geometries and hasattr(loaded, "faces"):
        geometries = [loaded]
    mask = np.zeros((height, width), dtype=np.uint8)
    for geometry in geometries:
        uv = getattr(getattr(geometry, "visual", None), "uv", None)
        faces = getattr(geometry, "faces", None)
        if uv is None or faces is None:
            continue
        uv_array = np.asarray(uv, dtype=np.float32)
        faces_array = np.asarray(faces, dtype=np.int32)
        if len(uv_array) == 0 or len(faces_array) == 0 or faces_array.max(initial=-1) >= len(uv_array):
            continue
        points = np.empty((len(uv_array), 2), dtype=np.float32)
        points[:, 0] = np.clip(uv_array[:, 0], 0.0, 1.0) * (width - 1)
        points[:, 1] = (1.0 - np.clip(uv_array[:, 1], 0.0, 1.0)) * (height - 1)
        contours = np.rint(points[faces_array]).astype(np.int32)
        for start in range(0, len(contours), 20_000):
            cv2.fillPoly(mask, list(contours[start : start + 20_000]), 255)
    return mask > 0


def trellis_pad_texture(image: Any, coverage: Any, padding: int) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    source = np.asarray(coverage, dtype=bool)
    if not source.any() or source.all() or padding <= 0:
        return image
    expanded = cv2.dilate(
        source.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=padding,
    ).astype(bool)
    padding_area = expanded & ~source
    if not padding_area.any():
        return image

    distance_input = (~source).astype(np.uint8)
    distance, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    del distance
    max_label = int(labels.max(initial=0))
    if max_label <= 0:
        return image
    channels = array.shape[2]
    palette = np.zeros((max_label + 1, channels), dtype=array.dtype)
    source_labels = labels[source]
    valid = source_labels > 0
    palette[source_labels[valid]] = array[source][valid]
    output = array.copy()
    target_labels = labels[padding_area]
    output[padding_area] = palette[target_labels]
    if output.shape[2] == 1:
        output = output[..., 0]
    return Image.fromarray(output, mode=image.mode)


def trellis_glb_stats(gltf: Any, blob: bytes, padding_report: dict[str, Any]) -> dict[str, Any]:
    import io
    from PIL import Image

    vertices = 0
    faces = 0
    has_uv = False
    for mesh in gltf.meshes or []:
        for primitive in mesh.primitives:
            attributes = primitive.attributes
            position = getattr(attributes, "POSITION", None)
            if position is not None:
                vertices += int(gltf.accessors[position].count)
            if getattr(attributes, "TEXCOORD_0", None) is not None:
                has_uv = True
            if primitive.indices is not None:
                faces += int(gltf.accessors[primitive.indices].count) // 3

    largest_texture = 0
    for image in gltf.images or []:
        if image.bufferView is None:
            continue
        view = gltf.bufferViews[image.bufferView]
        start = int(view.byteOffset or 0)
        try:
            with Image.open(io.BytesIO(blob[start : start + view.byteLength])) as texture:
                largest_texture = max(largest_texture, texture.width * texture.height)
        except Exception:
            continue
    coverages = [
        float(value["coverage"])
        for value in padding_report.values()
        if isinstance(value, dict) and isinstance(value.get("coverage"), (int, float))
    ]
    return {
        "vertices": vertices,
        "faces": faces,
        "hasUv": has_uv,
        "materials": len(gltf.materials or []),
        "textures": len(gltf.textures or []),
        "texelsPerFace": round(largest_texture / faces, 1) if largest_texture and faces else None,
        "atlasCoverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
    }


def normalize_trellis_glb(target: Path, options: dict[str, Any]) -> dict[str, Any]:
    """Normalize TRELLIS materials and pad UV atlas gutters.

    Atlas-padding behavior is adapted from dotneet/image-to-3d (MIT,
    Copyright (c) 2026 Shinji Yamada); see THIRD_PARTY_NOTICES.md.
    """
    import io

    from PIL import Image
    from pygltflib import BufferView, GLTF2

    gltf = GLTF2().load(str(target))
    changed_opaque = 0
    if as_bool(options.get("trellisForceOpaque"), True):
        for material in gltf.materials or []:
            if material.alphaMode != "OPAQUE":
                material.alphaMode = "OPAQUE"
                changed_opaque += 1

    padding = as_int(options.get("trellisAtlasPadding"), 12, 0, 64)
    padding_report: dict[str, Any] = {}
    blob = bytearray(gltf.binary_blob())
    if padding > 0 and gltf.images and gltf.textures:
        targets = {
            texture.source
            for texture in gltf.textures
            if texture.source is not None and texture.source < len(gltf.images)
        }
        masks: dict[tuple[int, int], Any] = {}
        for image_index in sorted(targets):
            image = gltf.images[image_index]
            if image.bufferView is None:
                continue
            view = gltf.bufferViews[image.bufferView]
            start = int(view.byteOffset or 0)
            raw = bytes(blob[start : start + view.byteLength])
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                texture = source.copy()
            key = texture.size
            coverage = masks.get(key)
            if coverage is None:
                coverage = trellis_uv_coverage_mask(target, texture.width, texture.height)
                masks[key] = coverage
            padded = trellis_pad_texture(texture, coverage, padding)
            mime_type = str(image.mimeType or "").lower()
            image_format = "WEBP" if mime_type == "image/webp" else "PNG"
            encoded = io.BytesIO()
            save_args = {"lossless": True, "quality": 95} if image_format == "WEBP" else {}
            padded.save(encoded, format=image_format, **save_args)
            data = encoded.getvalue()
            while len(blob) % 4:
                blob.append(0)
            offset = len(blob)
            blob.extend(data)
            gltf.bufferViews.append(
                BufferView(buffer=0, byteOffset=offset, byteLength=len(data))
            )
            image.bufferView = len(gltf.bufferViews) - 1
            image.mimeType = "image/webp" if image_format == "WEBP" else "image/png"
            padding_report[f"image_{image_index}"] = {
                "width": texture.width,
                "height": texture.height,
                "coverage": round(float(coverage.mean()), 4),
                "padding": padding,
            }

    while len(blob) % 4:
        blob.append(0)
    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(target))
    stats = trellis_glb_stats(gltf, bytes(blob), padding_report)
    stats.update(
        {
            "opaqueMaterialsNormalized": changed_opaque,
            "atlasPadding": padding,
            "atlas": padding_report,
        }
    )
    return stats


def trellis_pipeline_type(options: dict[str, Any]) -> str:
    resolution = as_int(options.get("trellisResolution"), 1024, 512, 1536)
    if resolution <= 512:
        return "512"
    if resolution >= 1536:
        return "1536_cascade"
    return "1024_cascade"


def trellis_sampler_params(options: dict[str, Any], prefix: str, defaults: tuple[int, float, float, float]) -> dict[str, Any]:
    default_steps, default_guidance, default_rescale, default_rescale_t = defaults
    return {
        "steps": as_int(options.get(f"{prefix}Steps"), default_steps, 1, 100),
        "guidance_strength": as_float(options.get(f"{prefix}Guidance"), default_guidance, 0, 30),
        "guidance_rescale": as_float(options.get(f"{prefix}Rescale"), default_rescale, 0, 2),
        "rescale_t": as_float(options.get(f"{prefix}RescaleT"), default_rescale_t, 0, 10),
    }


def trellis_export_glb(mesh: Any, target: Path, options: dict[str, Any]) -> dict[str, Any]:
    import o_voxel

    if hasattr(mesh, "simplify"):
        mesh.simplify(16_777_216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=as_int(options.get("trellisDecimationTarget"), 5_000, 1_000, 5_000_000),
        texture_size=as_int(options.get("trellisTextureSize"), 1024, 512, 8192),
        remesh=as_bool(options.get("trellisRemesh"), True),
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(target), extension_webp=as_bool(options.get("trellisExportWebp"), True))
    try:
        stats = normalize_trellis_glb(target, options)
    except Exception as error:
        emit("setup", f"TRELLIS.2 GLB post-processing skipped: {error}")
        stats = {
            "faces": int(len(getattr(mesh, "faces", []))),
            "atlasPadding": 0,
            "postprocessError": str(error),
        }
    if as_bool(options.get("smoothNormals"), True):
        try:
            stats["smoothedNormalPrimitives"] = maybe_smooth_exported_glb(target, options)
        except Exception as error:
            emit("setup", f"TRELLIS.2 smooth-normal export skipped: {error}")
            stats["smoothNormalsError"] = str(error)
    texels_per_face = stats.get("texelsPerFace")
    if isinstance(texels_per_face, (int, float)) and texels_per_face < 30:
        emit(
            "setup",
            f"TRELLIS.2 atlas density is low ({texels_per_face:.1f} texels per face). "
            "Use a larger texture or a lower face target if seams look blurred.",
        )
    return stats


def run_trellis_generation(job: dict[str, Any], paths: dict[str, Path]) -> None:
    source_dir = paths["trellis_source_dir"]
    launch_cwd = Path.cwd()
    patch_trellis_dinov3_source(source_dir)
    patch_trellis_attention_source(source_dir)
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not PIPELINE_CACHE.get("torch_configured"):
        configure_torch_performance(torch)
        PIPELINE_CACHE["torch_configured"] = True
    release_cached_pipelines(
        ("shape", "paint", "background"),
        torch,
        "Released Hunyuan3D models before TRELLIS.2 generation",
    )

    options = job.get("options", {})
    apply_trellis_preset(options)
    overwrite = as_bool(options.get("overwrite"), False)
    requested_output = Path(str(options.get("outputDir", "")).strip() or paths["output_dir"])
    if not requested_output.is_absolute():
        requested_output = (launch_cwd / requested_output).resolve()
    requested_output.mkdir(parents=True, exist_ok=True)
    cleanup_job_scratch(paths["jobs_dir"])
    run_dir = paths["jobs_dir"] / job.get("job_id", f"job-{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in job.get("paths", [])]
    total = len(input_paths)
    failed_count = 0
    batch_started = time.perf_counter()
    begin_job_manifest(job, paths, "trellis2")

    try:
        key = (TRELLIS_MODEL_REPO, as_bool(options.get("forceOffload"), False))
        pipeline = PIPELINE_CACHE.get("trellis_pipeline") if PIPELINE_CACHE.get("trellis_key") == key else None
        if pipeline is None:
            model_started = time.perf_counter()
            emit("model", "Loading TRELLIS.2 model on CUDA")
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_REPO)
            if as_bool(options.get("forceOffload"), False):
                pipeline.low_vram = True
            pipeline.cuda()
            PIPELINE_CACHE["trellis_pipeline"] = pipeline
            PIPELINE_CACHE["trellis_key"] = key
            emit_timing("TRELLIS.2 model load", model_started)
        else:
            emit("model", "Reusing loaded TRELLIS.2 model")

        for index, image_path in enumerate(input_paths):
            image_started = time.perf_counter()
            emit("image_start", f"Processing {image_path.name}", path=str(image_path))
            try:
                if not image_path.is_file():
                    raise RuntimeError(f"Input image was not found: {image_path}")
                base = output_base_name(options, image_path, index, total)
                final_mesh_path = unique_target(requested_output / f"{base}.glb", overwrite)
                image = Image.open(image_path)
                crop_info = None
                if as_bool(options.get("trellisAutocrop"), True):
                    crop_started = time.perf_counter()
                    emit("stage_start", f"Cropping subject for {image_path.name}", path=str(image_path))
                    try:
                        image, crop_info = trellis_autocrop_subject(pipeline, image)
                        if crop_info:
                            emit(
                                "setup",
                                f"Subject crop {crop_info['source'][0]}x{crop_info['source'][1]} "
                                f"to {crop_info['output'][0]}x{crop_info['output'][1]}.",
                                path=str(image_path),
                            )
                    except Exception as error:
                        crop_info = {"error": str(error)}
                        emit(
                            "setup",
                            f"TRELLIS.2 subject crop skipped: {error}",
                            path=str(image_path),
                        )
                    emit_timing("TRELLIS.2 subject crop", crop_started, path=str(image_path))
                seed = seed_value(options, "trellisSeed", "trellisSeedMode")
                stage_started = time.perf_counter()
                emit("stage_start", f"Generating TRELLIS.2 asset for {image_path.name}", path=str(image_path))
                meshes = pipeline.run(
                    image,
                    seed=seed,
                    pipeline_type=trellis_pipeline_type(options),
                    sparse_structure_sampler_params=trellis_sampler_params(
                        options,
                        "trellisSparse",
                        (12, 7.5, 0.7, 5.0),
                    ),
                    shape_slat_sampler_params=trellis_sampler_params(
                        options,
                        "trellisShape",
                        (12, 7.5, 0.5, 3.0),
                    ),
                    tex_slat_sampler_params=trellis_sampler_params(
                        options,
                        "trellisTexture",
                        (12, 3.0, 0.5, 3.0),
                    ),
                )
                emit_timing("TRELLIS.2 generation", stage_started, path=str(image_path))
                export_started = time.perf_counter()
                emit("stage_start", f"Exporting TRELLIS.2 GLB for {image_path.name}", path=str(image_path))
                glb_stats = trellis_export_glb(meshes[0], final_mesh_path, options)
                emit_timing("TRELLIS.2 GLB export", export_started, path=str(image_path))
                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_done",
                    f"Saved output in {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    outputs=[str(final_mesh_path.resolve())],
                    elapsedSeconds=round(elapsed_seconds, 3),
                    stats={
                        "preset": options.get("trellisPreset", "custom"),
                        "seed": seed,
                        "autocrop": crop_info,
                        "glb": glb_stats,
                    },
                )
            except Exception as error:
                failed_count += 1
                elapsed_seconds = time.perf_counter() - image_started
                emit(
                    "image_error",
                    f"{error} after {format_elapsed(elapsed_seconds)}",
                    path=str(image_path),
                    elapsedSeconds=round(elapsed_seconds, 3),
                    traceback=traceback.format_exc(),
                )

        batch_elapsed_seconds = time.perf_counter() - batch_started
        batch_elapsed = format_elapsed(batch_elapsed_seconds)
        if failed_count:
            emit(
                "done",
                f"Batch finished with errors in {batch_elapsed}.",
                failedCount=failed_count,
                totalCount=total,
                elapsedSeconds=round(batch_elapsed_seconds, 3),
            )
        else:
            emit(
                "done",
                f"Batch finished in {batch_elapsed}.",
                failedCount=0,
                totalCount=total,
                elapsedSeconds=round(batch_elapsed_seconds, 3),
            )
    finally:
        if not as_bool(options.get("keepModelsWarm"), True):
            release_cached_pipelines(("trellis",), torch, "Released TRELLIS.2 model after batch")
        shutil.rmtree(run_dir, ignore_errors=True)
        os.chdir(launch_cwd)


def run_generation(job: dict[str, Any], paths: dict[str, Path]) -> None:
    options = job.get("options", {})
    if normalize_model_backend(options.get("modelBackend")) == "trellis2":
        run_trellis_generation(job, paths)
        return
    run_hunyuan_generation(job, paths)


def run_managed_job(job: dict[str, Any]) -> None:
    app_data_dir = Path(job["app_data_dir"])
    options = job.get("options", {})
    pytorch_profile = normalize_pytorch_profile(options.get("runtimeProfile"))
    model_backend = normalize_model_backend(options.get("modelBackend"))
    paths = app_paths(app_data_dir, pytorch_profile)
    for key in ("runtime_dir", "model_dir", "output_dir", "jobs_dir", "trellis_extensions_dir", "cache_dir", "tmp_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    configure_portable_cache_env(paths)

    action = str(job.get("action", "generate"))
    configure_cuda_device(options.get("cudaDevice", "auto"))
    if model_backend == "trellis2":
        ensure_trellis_source(paths)
        ensure_trellis_models(paths)
    else:
        include_texture = action == "setup" or str(options.get("mode", "meshTexture")) != "meshOnly"
        if paths["hunyuan_source_dir"].is_dir():
            patch_hunyuan_source(paths["hunyuan_source_dir"])
        ensure_hunyuan_models(paths, include_texture=include_texture)

    if action == "setup":
        label = "TRELLIS.2" if model_backend == "trellis2" else "Hunyuan3D"
        emit("runtime_ready", f"Standalone {label} runtime is ready.")
        emit("done", f"Standalone {label} runtime is ready.")
        return

    run_generation(job, paths)


def run_outer(job_path: Path) -> int:
    with job_path.open("r", encoding="utf-8") as handle:
        job = json.load(handle)
    app_data_dir = Path(job["app_data_dir"])
    options = job.get("options", {})
    pytorch_profile = normalize_pytorch_profile(options.get("runtimeProfile"))
    model_backend = normalize_model_backend(options.get("modelBackend"))
    python_path, paths = prepare_managed_environment(app_data_dir, pytorch_profile, model_backend)
    return reinvoke_inside_env(python_path, job_path, app_data_dir, paths)


def run_daemon() -> int:
    if os.environ.get("I3D_IN_MANAGED_ENV") != "1" and os.environ.get("IH3D_IN_MANAGED_ENV") != "1":
        emit("error", "Managed worker daemon must be launched inside the app Python environment.")
        return 1
    emit("daemon_ready", "Managed Image To 3D worker is ready.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "__shutdown__":
            emit("daemon_exit", "Managed Image To 3D worker is shutting down.")
            return 0
        try:
            request = json.loads(line)
            job_path_value = request.get("jobPath") or request.get("job_path")
            if not job_path_value:
                raise RuntimeError("Daemon request did not include jobPath.")
            job_path = Path(str(job_path_value))
            with job_path.open("r", encoding="utf-8") as handle:
                job = json.load(handle)
            run_managed_job(job)
        except KeyboardInterrupt:
            emit("canceled", "Job canceled.")
        except Exception as error:
            emit("error", str(error), traceback=traceback.format_exc())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()

    try:
        if args.daemon:
            return run_daemon()
        if args.job is None:
            parser.error("--job is required unless --daemon is used")
        if os.environ.get("I3D_IN_MANAGED_ENV") == "1" or os.environ.get("IH3D_IN_MANAGED_ENV") == "1":
            with args.job.open("r", encoding="utf-8") as handle:
                job = json.load(handle)
            run_managed_job(job)
            return 0
        return run_outer(args.job)
    except KeyboardInterrupt:
        emit("canceled", "Job canceled.")
        return 130
    except Exception as error:
        emit("error", str(error), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
