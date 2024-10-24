import requests
import pandas as pd
import io
from .cache import cache
import xml.etree.cElementTree as ET
import tqdm.auto as tqdm


def get_datasets(mart: str = "ENSEMBL_MART_ENSEMBL", baseurl: str = "http://www.ensembl.org/biomart/martservice?") -> pd.DataFrame:
    """
    List all datasets available within a mart and baseurl
    """
    url = f"{baseurl}type=datasets&requestid=biomaRt&mart={mart}"
    if url in cache:
        cache[url]
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
    def __init__(self, name, value, **kwargs):
        self.name = name
        self.value = value
        self.kwargs = kwargs

    def to_xml(self):
        if isinstance(self.value, str):
            value = self.value
        else:
            if not all(isinstance(v, str) for v in self.value):
                raise ValueError("Filter value must be a string")
            value = ",".join(self.value)
        return ET.Element("Filter", name=self.name, value=value, **self.kwargs)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, ix):
        return Filter(self.name, **{k: v[ix] for k, v in self.kwargs.items()}, value=self.value[ix])


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
        """
        List all attributes available in a dataset
        """
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
        """
        List all filters available in a dataset
        """
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

    def get(self, attributes=[], filters=[], use_cache=True, timeout = 20) -> pd.DataFrame:
        """
        Get the result with a given set of attributes and filters

        If the result is already in the cache, it will be returned from the cache

        Parameters:
            attributes:
                list of attributes to return
            filters:
                list of filters to apply
            use_cache:
                whether to use the cache
        """

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

        if use_cache and (url in cache):
            result = cache[url]
        else:
            try:
                response = requests.get(url, timeout=timeout)
            except requests.exceptions.Timeout:
                raise ValueError("Ensembl web service timed out")
            # check response status
            if response.status_code != 200:
                raise ValueError(f"Response status code is {response.status_code} and not 200. Response text: {response.text}")
            if "Query ERROR: caught BioMart" in response.text:
                print(query.replace("><", ">\n<"))
                raise ValueError(response.text)
            if "The Ensembl web service you requested is temporarily unavailable." in response.text:
                raise ValueError("Ensembl web service is temporarily unavailable")
            result = pd.read_table(
                io.StringIO(response.text),
                sep="\t",
                names=[attribute.name for attribute in attributes],
            )
            cache[url] = result
        return result

    def get_batched(self, attributes=[], filters=[], batch_size=50, use_cache=True) -> pd.DataFrame:
        """
        Get the result with a given set of attributes and filters, but batched

        Parameters:
            attributes:
                list of attributes to return
            filters:
                list of filters to apply
            batch_size:
                batch size
        """
        assert len(filters) == 1
        filter = filters[0]

        result = []
        for i in tqdm.tqdm(range(0, len(filter), batch_size), leave=False):
            filters_ = [filter[i : i + batch_size]]
            result.append(self.get(attributes=attributes, filters=filters_, use_cache=use_cache))
        result = pd.concat(result)
        return result

    @classmethod
    def from_genome(self, genome):
        """
        Get the biomart dataset given a particular genome name, e.g. GRCm38, GRCh38, GRCm39, mm10, hg19, ...
        """
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
                "http://www.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        elif genome in ["GRCz11"]:
            return Dataset(
                "drerio_gene_ensembl",
                "http://www.ensembl.org/biomart/martservice?",
                "ENSEMBL_MART_ENSEMBL",
            )
        else:
            raise ValueError("Genome not supported")
