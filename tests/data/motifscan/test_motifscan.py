import pandas as pd
import chromatinhd as chd
import numpy as np
import torch


def test_create_onehots():
    assert (
        chd.data.motifscan.motifscan.create_onehots(["ACGTN", "GCTNA", "NNNNN"])
        == torch.tensor(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ]
        )
    ).all()


def test_scan():
    pwm = torch.tensor(
        [
            [1.0, 0, 0, 0],
            [0.0, 1, 0, 0],
            [0.0, 0, 1, 0],
        ]
    ).T
    tests = [
        {
            "sequence": "ACGT",
            "pwm": pwm,
            "expected_scores": np.array([3.0, 3.0]),
            "expected_positions": np.array([0, 1]),
            "expected_strands": np.array([1, -1]),
        },
        {
            "sequence": "AAAA",
            "pwm": pwm,
            "expected_scores": np.array([]),
            "expected_positions": np.array([]),
            "expected_strands": np.array([]),
        },
        {
            "sequence": "CGT",
            "pwm": pwm,
            "expected_scores": np.array([3.0]),
            "expected_positions": np.array([0]),
            "expected_strands": np.array([-1]),
        },
        {
            "sequence": "CGTCGT",
            "pwm": pwm,
            "expected_scores": np.array([2.0, 3.0, 3.0]),
            "expected_positions": np.array([2, 0, 3]),
            "expected_strands": np.array([1, -1, -1]),
        },
    ]

    for test in tests:
        onehot = chd.data.motifscan.motifscan.create_onehots([test["sequence"]]).permute(0, 2, 1)
        pwm = test["pwm"]

        scores, positions, strands = chd.data.motifscan.motifscan.scan(onehot, pwm, cutoff=1.5)

        assert np.allclose(scores, test["expected_scores"]), (test["sequence"], scores)
        assert np.allclose(positions, test["expected_positions"].astype(positions.dtype))
        assert np.allclose(strands, test["expected_strands"].astype(np.int8))


class TestMotifscan:
    def test_simple(self, tmp_path):
        # create sequence
        sequence_1 = "CGTT"
        sequence_2 = "ACGA"
        sequence_3 = "AACG"

        fa_file = f"{tmp_path}/test.fa"
        with open(fa_file, "w", encoding="utf8") as f:
            f.write(">chr1\n")
            f.write(sequence_1 + "\n")
            f.write(">chr2\n")
            f.write(sequence_2 + "\n")
            f.write(">chr3\n")
            f.write(sequence_3 + "\n")

        region_size = len(sequence_1)
        # create regions
        region_coordinates = pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr3"],
                "start": [0, 0, 0],
                "end": [region_size, region_size, region_size],
                "tss": [0, 0, 0],
                "strand": [1, 1, -1],
            }
        )
        regions = chd.data.regions.Regions.create(
            path=f"{tmp_path}/regions",
            coordinates=region_coordinates,
            window=(0, region_size),
        )

        # create motifs
        pwms = {
            "A": torch.tensor(
                [
                    [1.0, 0, 0, 0],
                    [1.0, 0, 0, 0],
                ]
            ),
            "B": torch.tensor(
                [
                    [1.0, 0, 0, 0],
                    [0.0, 1, 0, 0],
                    [0.0, 0, 1, 0],
                ]
            ),
        }
        motifs = pd.DataFrame(
            {
                "motif": ["A", "B"],
            }
        ).set_index("motif")
        cutoffs = [1.5, 1.5]
        motifs["cutoff"] = cutoffs

        motifscan = chd.data.motifscan.Motifscan.from_pwms(
            pwms=pwms,
            regions=regions,
            fasta_file=fa_file,
            cutoffs=motifs["cutoff"],
            path=f"{tmp_path}/motifscan",
            min_cutoff=0.0,
        )

        assert np.array_equal(
            motifscan.coordinates[:],
            np.array([0, 2, 0, 1, 0, 2], dtype=int),
        )
        assert np.array_equal(
            motifscan.scores[:],
            np.array([3.0, 2.0, 3.0, 2.0, 3.0, 2.0]),
        )
        assert np.array_equal(
            motifscan.strands[:],
            np.array([-1, -1, 1, -1, -1, -1], dtype=np.int8),
        )
        assert np.array_equal(
            motifscan.region_indices[:],
            np.array([0, 0, 1, 1, 2, 2], dtype=np.int8),
        )

        motifscan.create_region_indptr()
        motifscan.create_indptr()
        assert len(motifscan.indptr) == (region_size * len(region_coordinates) + 1)

        # get all sites
        motifscan = motifscan.from_pwms(
            pwms=pwms,
            regions=regions,
            fasta_file=fa_file,
            cutoffs=-99999,
            min_cutoff=-9999,
            path=f"{tmp_path}/motifscan",
        )
        motifscan.create_region_indptr()
        motifscan.create_indptr()
        assert len(motifscan.indptr) == (region_size * len(region_coordinates) + 1)
