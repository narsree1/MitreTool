# utils.py - NO TORCH DEPENDENCIES
import requests

def load_lottie_url(url: str):
    """
    Load a Lottie animation from a URL
    
    Args:
        url: URL to the Lottie JSON file
        
    Returns:
        Lottie animation JSON or None if failed
    """
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None
