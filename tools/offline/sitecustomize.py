"""Test-process-only network guard, inherited by Python subprocesses."""
import ipaddress
import os
import sys

if os.environ.get("APS_OFFLINE_TESTS") == "1":
    for key in list(os.environ):
        if key.endswith(("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN")) or key in {"OPENAI_ADMIN_KEY", "HF_TOKEN"}:
            os.environ.pop(key, None)

    class NetworkForbidden(BaseException):
        pass

    def loopback(host):
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="replace")
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False

    def audit(event, args):
        if event == "socket.getaddrinfo":
            host = args[0]
        elif event in {"socket.connect", "socket.sendto"}:
            address = args[1] if event == "socket.connect" else args[-1]
            host = address[0] if isinstance(address, tuple) else address
        else:
            return
        if not loopback(host):
            raise NetworkForbidden(f"Offline test attempted external network: {event} {host!r}")

    sys.addaudithook(audit)
    sys._aps_network_guard = True
