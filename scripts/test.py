# %%
import requests

# Send a GET request to the BioMart REST API
response = requests.get('http://www.ensembl.org/biomart/martservice?type=registry')

# The response is an XML string, so parse it using ElementTree
import xml.etree.ElementTree as ET
root = ET.fromstring(response.text)

# Iterate over all the MartURLLocation elements (these represent the datasets)
for dataset in root.iter('MartURLLocation'):
    # The species is stored in the 'name' attribute
    species = dataset.get('name')

    # If the species is human, print the details of the dataset
    if 'hsapiens' in species:
        print(ET.tostring(dataset, encoding='utf8').decode('utf8'))

# %%
response.text
# %%
import pandas as pd
import io

def get_datasets():
    mart = "ENSEMBL_MART_ENSEMBL"
    baseurl = "http://www.ensembl.org/biomart/martservice?"
    url = "{baseurl}type=datasets&requestid=biomaRt&mart={mart}"
    response = requests.get(url.format(baseurl=baseurl, mart=mart))
    root = pd.read_table(io.StringIO(response.text), sep="\t", header=None, names=["_", "dataset", "description", "version", "assembly", "__", "___", "____", "last_update"])
    print(root)
# %%
get_biomart_datasets()
# %%
