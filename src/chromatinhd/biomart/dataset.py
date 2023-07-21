import requests
import pandas as pd
import io
from .cache import cache
import xml.etree.cElementTree as ET


def get_datasets(
    mart="ENSEMBL_MART_ENSEMBL", baseurl="http://www.ensembl.org/biomart/martservice?"
):
    url = f"{baseurl}type=datasets&requestid=biomaRt&mart={mart}"
    if url in cache:
        attributes = cache[url]
    else:
        response = requests.get(url)
        datasets = pd.read_table(
            io.StringIO(response.text),
            sep="\t",
            header=None,
            names=[
                "_",
                "dataset",
                "description",
                "version",
                "assembly",
                "__",
                "___",
                "____",
                "last_update",
            ],
        )
        cache[url] = datasets
    return datasets


class Attribute:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def to_xml(self):
        return ET.Element("Attribute", name=self.name, **self.kwargs)


class Filter:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def to_xml(self):
        return ET.Element("Filter", name=self.name, **self.kwargs)


class Dataset:
    def __init__(
        self,
        name="hsapiens_gene_ensembl",
        baseurl="http://www.ensembl.org/biomart/martservice?",
        mart="ENSEMBL_MART_ENSEMBL",
    ):
        self.name = name
        self.mart = mart
        self.baseurl = baseurl

    def list_attributes(self):
        url = f"{self.baseurl}type=attributes&dataset={self.name}&mart={self.mart}"

        if url in cache:
            attributes = cache[url]
        else:
            response = requests.get(url)
            attributes = pd.read_table(
                io.StringIO(response.text),
                sep="\t",
                header=None,
                names=[
                    "attribute",
                    "name",
                    "description",
                    "type",
                    "format",
                    "full_attribute",
                    "alternative_attribute",
                ],
            ).set_index("attribute")
            cache[url] = attributes
        return attributes

    def list_filters(self):
        url = f"{self.baseurl}type=filters&dataset={self.name}&mart={self.mart}"

        if url in cache:
            filters = cache[url]
        else:
            response = requests.get(url)
            filters = pd.read_table(
                io.StringIO(response.text),
                sep="\t",
                header=None,
                names=[
                    "filter",
                    "name",
                    "options",
                    "description",
                    "_",
                    "type",
                    "operation",
                    "full_filter",
                    "alternative_filter",
                ],
            ).set_index("filter")
            cache[url] = filters
        return filters

    def attribute(self, name, **kwargs):
        return Attribute(name, **kwargs)

    def filter(self, name, **kwargs):
        return Filter(name, **kwargs)

    def get(self, attributes=[], filters=[]):
        xml = ET.Element(
            "Query",
            virtualSchemaName="default",
            formatter="TSV",
            header="0",
            uniqueRows="0",
            datasetConfigVersion="0.6",
            count="",
        )
        dataset = ET.SubElement(xml, "Dataset", name=self.name, interface="default")
        for filter in filters:
            dataset.append(filter.to_xml())
        for attribute in attributes:
            dataset.append(attribute.to_xml())
        query = ET.tostring(xml).decode("utf-8")
        query = query.replace("\t", "").replace("\n", "")
        url = f"{self.baseurl}query={query}"

        if url in cache:
            result = cache[url]
        else:
            response = requests.get(url)
            result = pd.read_table(
                io.StringIO(response.text),
                sep="\t",
                names=[attribute.name for attribute in attributes],
            )
            cache[url] = result
        return result

    @classmethod
    def from_genome(self, genome):
        if genome in ["mm10", "GRCm38"]:
            return Dataset(
                "mmusculus_gene_ensembl",
                "https://nov2020.archive.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        elif genome in ["hg19", "GRCh37"]:
            return Dataset(
                "hsapiens_gene_ensembl",
                "http://grch37.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        elif genome in ["hg38", "GRCh38"]:
            return Dataset(
                "hsapiens_gene_ensembl",
                "http://www.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        elif genome in ["GRCm39"]:
            return Dataset(
                "mmusculus_gene_ensembl",
                "https://nov2020.archive.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        else:
            raise ValueError("Genome not supported")
