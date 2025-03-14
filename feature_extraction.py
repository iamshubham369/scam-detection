import re
import whois
import datetime
import requests
from urllib.parse import urlparse

def extract_features(url):
    features = []

    # 1. URL Length
    features.append(len(url))

    # 2. Presence of '@' symbol
    features.append(1 if '@' in url else 0)

    # 3. Presence of '-'
    features.append(1 if '-' in urlparse(url).netloc else 0)

    # 4. Number of dots in domain
    features.append(url.count('.'))

    # 5. HTTPS usage
    features.append(1 if url.startswith('https') else 0)

    # 6. Presence of 'https' in domain name (bad practice)
    features.append(1 if 'https' in urlparse(url).netloc else 0)

    # 7. WHOIS domain age
    try:
        domain_info = whois.whois(urlparse(url).netloc)
        creation_date = domain_info.creation_date
        expiration_date = domain_info.expiration_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        age = (datetime.datetime.now() - creation_date).days if creation_date else 0
        features.append(age)
    except:
        features.append(0)

    # 8. Website response (checking if active)
    try:
        response = requests.get(url, timeout=5)
        features.append(1 if response.status_code == 200 else 0)
    except:
        features.append(0)

    # Add 22 more dummy features (replace these with real ones used in training)
    for i in range(22):
        features.append(0)  # Placeholder for missing features

    return features
