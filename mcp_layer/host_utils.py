import requests


def get_imds_token() -> str | None:
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    try:
        response = requests.put(token_url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error getting IMDS token: {e}")
        return None


def get_metadata_with_token(path: str, token: str) -> str | None:
    url = f"http://169.254.169.254/latest/meta-data/{path}"
    headers = {"X-aws-ec2-metadata-token": token}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching metadata for {path}: {e}")
        return None


def resolve_host() -> str:
    """Resolve the public IP via IMDSv2, falling back to localhost."""
    token = get_imds_token()
    if token:
        host = get_metadata_with_token("public-ipv4", token)
        if host:
            return host
    print("Could not obtain IMDSv2 token — falling back to localhost.")
    return "localhost"