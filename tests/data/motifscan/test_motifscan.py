import pandas as pd
import chromatinhd as chd
import numpy as np
import torch


def test_digitize_sequence():
    assert (
        chd.data.motifscan.motifscan.digitize_sequence("ACGTN")
        == np.array([0, 1, 2, 3, 4])
    ).all()


def test_create_onehot():
    assert (
        chd.data.motifscan.motifscan.create_onehot(
            chd.data.motifscan.motifscan.digitize_sequence("ACGTN")
        )
        == torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
    ).all()


def test_scan():
    pwm = torch.tensor(
        [
            [1.0, 0, 0, 0],
            [0.0, 1, 0, 0],
            [0.0, 0, 1, 0],
        ]
    )
    tests = [
        {
            "sequence": "ACGT",
            "pwm": pwm,
            "expected_scores": torch.tensor([3.0, 3.0]),
            "expected_positions": torch.tensor([[0, 0], [0, 3]]),
            "expected_strands": torch.tensor([1, -1]),
        },
        {
            "sequence": "AAAA",
            "pwm": pwm,
            "expected_scores": torch.tensor([]),
            "expected_positions": torch.tensor([]).reshape(2, 0),
            "expected_strands": torch.tensor([]),
        },
        {
            "sequence": "CGT",
            "pwm": pwm,
            "expected_scores": torch.tensor([3.0]),
            "expected_positions": torch.tensor([[0], [2]]),
            "expected_strands": torch.tensor([-1]),
        },
        {
            "sequence": "CGTCGT",
            "pwm": pwm,
            "expected_scores": torch.tensor([2.0, 3.0, 3.0]),
            "expected_positions": torch.tensor([[0, 0, 0], [2, 2, 5]]),
            "expected_strands": torch.tensor([1, -1, -1]),
        },
    ]

    for test in tests:
        onehot = chd.data.motifscan.motifscan.create_onehot(
            chd.data.motifscan.motifscan.digitize_sequence(test["sequence"])
        )[None, ...]
        pwm = test["pwm"]

        scores, positions, strands = chd.data.motifscan.motifscan.scan(
            onehot, pwm, cutoff=1.5
        )

        assert torch.equal(scores, test["expected_scores"]), (test["sequence"], scores)
        assert torch.equal(positions, test["expected_positions"].to(torch.int))
        assert torch.equal(strands, test["expected_strands"].to(torch.int8))


class TestMotifscan:
    def test_simple(self, tmp_path):
        # create sequence
        sequence_1 = "CGTT"
        sequence_2 = "ACGA"
        sequence_3 = "ACAA"

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
                "tss": [1, 1, 1],
                "strand": [1, 1, -1],
            }
        )
        regions = chd.data.regions.Regions.create(
            path=f"{tmp_path}/regions",
            coordinates=region_coordinates,
        )

        motifscan = chd.data.motifscan.Motifscan(f"{tmp_path}/motifscan")

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

        motifscan = motifscan.from_pwms(
            pwms=pwms,
            regions=regions,
            fasta_file=fa_file,
            cutoffs=motifs["cutoff"],
            path=f"{tmp_path}/motifscan",
        )

        assert np.array_equal(
            motifscan.positions,
            np.array([2, 3, 4, 7, 8, 10], dtype=int),
        )
        assert np.array_equal(
            motifscan.scores,
            np.array([3.0, 2.0, 3.0, 2.0, 2.0, 2.0]),
        )
        assert np.array_equal(
            motifscan.strands,
            np.array([-1, -1, 1, -1, 1, 1], dtype=np.int8),
        )
        assert len(motifscan.indptr) == (region_size * len(region_coordinates) + 1)

        # get all sites
        motifscan = motifscan.from_pwms(
            pwms=pwms,
            regions=regions,
            fasta_file=fa_file,
            cutoffs=-99999,
            path=f"{tmp_path}/motifscan",
        )
        assert len(motifscan.indptr) == (region_size * len(region_coordinates) + 1)
