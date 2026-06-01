from __future__ import annotations

import json

import pandas as pd


NGUNI_DOCUMENT_CANONICALS: dict[str, dict[str, str]] = {
    "xho": {
        "identity_docs": "isazisi",
        "certificate": "isiqinisekiso",
        "birth_certificate": "isatifikethi sokuzalwa",
        "death_certificate": "isatifikethi sokusweleka",
        "marriage_certificate": "isatifikethi somtshato",
        "medical_certificate": "isatifikethi sikagqirha",
        "police_clearance": "isatifikethi sokungabi namatyala",
        "tax_certificate": "isatifikethi serhafu",
        "permit": "iphepha-mvume",
        "work_permit": "iphepha-mvume lokusebenza",
        "study_permit": "iphepha-mvume lokufunda",
        "residence_permit": "iphepha-mvume lokuhlala",
        "passport": "ipasipoti",
        "drivers_license": "ilayisensi yokuqhuba",
        "proof_of_residence": "ubungqina bendawo yokuhlala",
        "application_form": "ifomu yesicelo",
        "registration_form": "ifomu yobhaliso",
        "affidavit": "ingxelo efungelweyo",
        "certified_copy": "ikopi eqinisekisiweyo",
        "social_grants": "isibonelelo",
        "grant_application": "isicelo sesibonelelo",
        "tax_number": "inombolo yerhafu",
        "bank_statement": "isiteyitimenti sebhanki",
        "proof_of_income": "ubungqina bengeniso",
        "school_report": "ingxelo yesikolo",
    },
    "zul": {
        "identity_docs": "umazisi",
        "certificate": "isitifiketi",
        "birth_certificate": "isitifiketi sokuzalwa",
        "death_certificate": "isitifiketi sokufa",
        "marriage_certificate": "isitifiketi somshado",
        "medical_certificate": "isitifiketi sikadokotela",
        "police_clearance": "isitifiketi sokuhlanzeka emaphoyiseni",
        "tax_certificate": "isitifiketi sentela",
        "permit": "imvume",
        "work_permit": "imvume yokusebenza",
        "study_permit": "imvume yokufunda",
        "residence_permit": "imvume yokuhlala",
        "passport": "iphasiphothi",
        "drivers_license": "ilayisensi yokushayela",
        "proof_of_residence": "ubufakazi bendawo yokuhlala",
        "application_form": "ifomu lesicelo",
        "registration_form": "ifomu lokubhalisa",
        "affidavit": "i-afidavithi",
        "certified_copy": "ikhophi eqinisekisiwe",
        "social_grants": "isibonelelo",
        "grant_application": "isicelo sesibonelelo",
        "tax_number": "inombolo yentela",
        "bank_statement": "isitatimende sasebhange",
        "proof_of_income": "ubufakazi bemali engenayo",
        "school_report": "umbiko wesikole",
    },
    "ssw": {
        "identity_docs": "mazisi",
        "certificate": "sitifiketi",
        "birth_certificate": "sitifiketi sekutalwa",
        "death_certificate": "sitifiketi sekufa",
        "marriage_certificate": "sitifiketi semshado",
        "medical_certificate": "sitifiketi sadokotela",
        "police_clearance": "sitifiketi sekungabi nemacala emaphoyiseni",
        "tax_certificate": "sitifiketi semtselo",
        "permit": "imvume",
        "work_permit": "imvume yekusebenza",
        "study_permit": "imvume yekufundza",
        "residence_permit": "imvume yekuhlala",
        "passport": "phasiphothi",
        "drivers_license": "ilayisensi yekushayela",
        "proof_of_residence": "bufakazi bendzawo yekuhlala",
        "application_form": "lifomu lesicelo",
        "registration_form": "lifomu lekubhalisa",
        "affidavit": "i-afidavithi",
        "certified_copy": "ikhophi lecinisekisiwe",
        "social_grants": "sibonelelo",
        "grant_application": "sicelo sesibonelelo",
        "tax_number": "inombolo yemtselo",
        "bank_statement": "sitatimende sasebhange",
        "proof_of_income": "bufakazi bemali lengenako",
        "school_report": "umbiko wesikolo",
    },
}


def _safe_json_list(value) -> list[str]:
    if isinstance(value, list):
        return value

    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    return json.loads(text)


def _normalise_space(text: str) -> str:
    return " ".join(str(text).split())


def _has_nguni_canonical(lang: str, semantic_type: str) -> bool:
    return (
        lang in NGUNI_DOCUMENT_CANONICALS
        and semantic_type in NGUNI_DOCUMENT_CANONICALS[lang]
    )


def normalise_nguni_document_inventory(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Promote Nguni document fragments to fuller canonical document phrases.

    Why this exists:
    - N-gram induction often finds useful but fragmentary Nguni forms.
      Examples:
        xho: sokuzalwa, sokusweleka, somtshato
        zul: sokuzalwa, sokufa, somshado
    - Those fragments are useful for matching source sentences.
    - They are poor as replacement surfaces.

    This function keeps the induced fragment as a matchable variant, but sets a
    fuller document phrase as the canonical replacement surface.
    """
    if inv_df.empty:
        return inv_df.copy()

    out = inv_df.copy()

    for idx, row in out.iterrows():
        lang = str(row.get("lang", ""))
        kind = str(row.get("kind", ""))
        semantic_type = str(row.get("semantic_type", ""))

        if kind != "document":
            continue

        if not _has_nguni_canonical(lang, semantic_type):
            continue

        canonical = NGUNI_DOCUMENT_CANONICALS[lang][semantic_type]
        old_canonical = _normalise_space(str(row.get("canonical_surface", "")))
        variants = _safe_json_list(row.get("swappable_variants_json", "[]"))

        all_variants = [canonical, old_canonical] + variants
        all_variants = [
            _normalise_space(v)
            for v in all_variants
            if str(v).strip()
        ]

        deduped = list(dict.fromkeys(all_variants))

        out.at[idx, "canonical_surface"] = canonical
        out.at[idx, "swappable_variants_json"] = json.dumps(deduped, ensure_ascii=False)
        out.at[idx, "swappable_variant_count"] = len(deduped)

    return out