_cache = {}


def get(query):
    global _cache
    if query in _cache:
        return _cache[query]

    from io import StringIO
    import requests
    import pandas as pd

    url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace(
        "\t", ""
    ).replace("\n", "")

    session = requests.Session()
    session.headers.update({"User-Agent": "Custom user agent"})
    r = session.get(url)

    if "Service unavailable" in r.content.decode("utf-8"):
        raise ValueError("Service unavailable")
    result = pd.read_table(StringIO(r.content.decode("utf-8")))

    _cache[query] = result
    return result
