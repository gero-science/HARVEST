"""Tests for IUPAC normalization functions in chemistry_rdkit.iupac."""

from chemistry_rdkit import (
    normalize_iupac_spelling, is_balanced,
)


class TestLocantLetterHyphen:
    """Test fix for erroneous hyphen between locant number and letter.

    Example: "4,4-a,5,6-tetrahydro" -> "4,4a,5,6-tetrahydro"
    """

    def test_epiminoethano_benzo_isoquinolin_methoxy(self):
        """Original case: methoxy variant"""
        original = "(4aS,5R,10bS)-13-(cyclopropylmethyl)-9-methoxy-3-(4-methoxybenzyl)-4,4-a,5,6-tetrahydro-1H-5,10b-(epiminoethano)benzo[f]isoquinolin-2(3H)-one"
        normalized = normalize_iupac_spelling(original)

        assert "4,4a,5,6" in normalized
        assert "4,4-a,5,6" not in normalized

    def test_epiminoethano_benzo_isoquinolin_hydroxy(self):
        """Hydroxy variant"""
        original = "2-((4aS,5R,10bR)-13-(cyclopropylmethyl)-9-hydroxy-4,4-a,5,6-tetrahydro-1H-5,10b-(epiminoethano)benzo[f]isoquinolin-3(2H)-yl)acetic acid"
        normalized = normalize_iupac_spelling(original)

        assert "4,4a,5,6" in normalized
        assert "4,4-a,5,6" not in normalized

    def test_epiminoethano_benzo_isoquinolin_propanone(self):
        """Hydroxy variant with propan-1-one suffix"""
        original = "(S)-2-amino-1-((4aS,5R,10bS)-13-(cyclopropylmethyl)-9-hydroxy-4,4-a,5,6-tetrahydro-1H-5,10b-(epiminoethano)benzo[f]isoquinolin-3(2H)-yl)propan-1-one"
        normalized = normalize_iupac_spelling(original)

        assert "4,4a,5,6" in normalized
        assert "4,4-a,5,6" not in normalized


class TestHeterocyclicAminoPosition:
    """Test fix for incorrect substituent position in heterocyclic amino groups.

    Example: "2-dibenzofuranamino" -> "dibenzofuran-2-ylamino"
    """

    def test_dibenzofuranamino(self):
        """Fix 2-dibenzofuranamino -> dibenzofuran-2-ylamino"""
        original = "2-((1R,2S)-2-aminocyclohexylamino)-4-(2-dibenzofuranamino)pyrimidine-5-carboxamide"
        normalized = normalize_iupac_spelling(original)

        assert "dibenzofuran-2-ylamino" in normalized
        assert "2-dibenzofuranamino" not in normalized

    def test_carbazolamino(self):
        """Test carbazol variant"""
        original = "4-(3-carbazolamino)benzamide"
        normalized = normalize_iupac_spelling(original)

        assert "carbazol-3-ylamino" in normalized
        assert "3-carbazolamino" not in normalized


class TestFusedRingAzolSuffix:
    """Test fix for fused ring -azol suffix before brackets.

    Example: "imidazol[4,5-c]" -> "imidazo[4,5-c]"
    """

    def test_imidazol_pyridazine_cyclobutyl(self):
        """Fix imidazol[ -> imidazo["""
        original = "7-Cyclobutyl-4-(6-fluoro-2'-methoxy-4'-(methylsulfonyl)-[1,1'-biphenyl]-3-yl)-7H-imidazol[4,5-c]pyridazine"
        normalized = normalize_iupac_spelling(original)

        assert "imidazo[4,5-c]" in normalized
        assert "imidazol[4,5-c]" not in normalized

    def test_imidazol_pyridazine_propanyl(self):
        """Another imidazol variant"""
        original = "4-[4'-(Ethylsulfonyl)-2',6-difluorobiphenyl-3-yl]-7-(propan-2-yl)-7H-imidazol[4,5-c]pyridazine"
        normalized = normalize_iupac_spelling(original)

        assert "imidazo[4,5-c]" in normalized
        assert "imidazol[4,5-c]" not in normalized

    def test_thiazol_fused(self):
        """Test thiazol -> thiazo"""
        original = "2-methyl-5H-thiazol[3,2-a]pyrimidine"
        normalized = normalize_iupac_spelling(original)

        assert "thiazo[3,2-a]" in normalized
        assert "thiazol[3,2-a]" not in normalized

    def test_oxazol_fused(self):
        """Test oxazol -> oxazo"""
        original = "6-bromo-2H-oxazol[4,5-b]pyridine"
        normalized = normalize_iupac_spelling(original)

        assert "oxazo[4,5-b]" in normalized
        assert "oxazol[4,5-b]" not in normalized


class TestFusedRingBracketInsertion:
    """Test insertion of brackets for fused ring notation.

    Example: "pyrido3,2-bpyrazin" -> "pyrido[3,2-b]pyrazin"
    """

    def test_pyrido_pyrazin(self):
        """Insert brackets for pyrido-pyrazin fusion"""
        original = "pyrido3,2-bpyrazine"
        normalized = normalize_iupac_spelling(original)

        assert "pyrido[3,2-b]pyrazine" in normalized

    def test_imidazo_pyridin_inside_long_name(self):
        """Insert brackets for imidazo-pyridin fusion inside a substituent"""
        original = "(1-(3-(tert-Butyl)-1-(4-methoxyphenyl)-1H-pyrazol-5-yl)-3-(4-(1-isopropyl-2-oxo-2,3-dihydro-1H-imidazo4,5-bpyridin-7-yl)oxy)naphthalen-1-yl)urea"
        normalized = normalize_iupac_spelling(original)

        assert "imidazo[4,5-b]pyridin" in normalized
        assert "imidazo4,5-b" not in normalized


class TestStereoisomerAlternatives:
    """Test handling of 'and' / 'or' between stereodescriptor sets."""

    def test_spaced_and_is_removed(self):
        normalized = normalize_iupac_spelling("(1S,2S,4R and 1R,2R,4S)-2-cyano-compound")

        assert "and" not in normalized.lower()

    def test_glued_and_is_preserved(self):
        normalized = normalize_iupac_spelling("(1S,2S,4Rand)-2-cyano-compound")

        assert "and" in normalized.lower()

    def test_spaced_or_is_removed_from_descriptor(self):
        normalized = normalize_iupac_spelling("(1S or 1R)-2,2,2-trifluoro-compound")

        assert normalized.lower().find("or") > 10

    def test_glued_or_is_preserved(self):
        normalized = normalize_iupac_spelling("(1Sor1R)-2,2,2-trifluoro-compound")

        assert "or" in normalized.lower()


class TestSafetyPreservation:
    """Test that dangerous transformations are NOT applied.

    These tests ensure we don't accidentally change molecular identity.
    """

    def test_stereoisomer_mixture_selects_first_branch(self):
        """Parenthetical stereoisomer alternatives select the first branch."""
        original = "{(1S,2S,4R and 1R,2R,4S)-2-cyano-1-methyl}"
        normalized = normalize_iupac_spelling(original)

        assert "1S,2S,4R" in normalized
        assert "1R,2R,4S" not in normalized

    def test_doubled_digits_preserved(self):
        """Doubled digits should NOT be modified.

        '4,4-dimethyl' means two methyl groups at position 4.
        """
        original = "4,4-dimethylcyclohexane"
        normalized = normalize_iupac_spelling(original)

        assert "4,4-dimethyl" in normalized

    def test_correct_fused_name_unchanged(self):
        """A correctly written fused-ring name is returned as is."""
        original = "7H-imidazo[4,5-c]pyridazine"

        assert normalize_iupac_spelling(original) == original


class TestTrailingCharacters:
    """Test removal of trailing periods and whitespace."""

    def test_trailing_period(self):
        """Remove trailing period"""
        original = "1-(3-isopropyl-1H-pyrazol-5-yl)urea."
        normalized = normalize_iupac_spelling(original)

        assert normalized == "1-(3-isopropyl-1H-pyrazol-5-yl)urea"

    def test_outer_parentheses(self):
        """Remove unnecessary outer parentheses"""
        original = "(benzene)"
        normalized = normalize_iupac_spelling(original)

        assert normalized == "benzene"


class TestBracketBalancing:
    """Test bracket balancing functions."""

    def test_balanced(self):
        """Already balanced"""
        assert len(is_balanced("(a[b]c)")) == 0

    def test_unbalanced_missing_close(self):
        """Missing closing bracket"""
        assert len(is_balanced("(a[b]c")) > 0

    def test_unbalanced_missing_open(self):
        """Missing opening bracket"""
        assert len(is_balanced("a[b]c)")) > 0
