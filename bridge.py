"""Bridge between NADI and FHIR R4 servers such as SATUSEHAT.

Three parts: claims as FHIR in both directions, the assessment as FHIR, and a small
client that posts to SATUSEHAT or to any other FHIR server.
"""
import copy
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import nadi

# AI Transparency on FHIR (hl7.fhir.uv.aitransparency, STU1 ballot, FHIR R4)
AI_IG = "http://hl7.org/fhir/uv/aitransparency"
AIAST = {"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue", "code": "AIAST",
         "display": "Artificial Intelligence asserted"}
AI_TYPE = {"coding": [{"system": AI_IG + "/CodeSystem/AIdeviceTypeCS", "code": "Artificial-Intelligence"}]}
# the sample data has no real identifiers; production would carry the SEP number and the IHS patient id
BASE = "https://github.com/AliifRahman/nadi/fhir/"
ICD10 = "http://hl7.org/fhir/sid/icd-10"
# no public canonical URL was found for INA-CBG codes or for the hospital class
INA_CBG = BASE + "ina-cbg"
HOSPITAL_CLASS = BASE + "kelas-rs"
PRINCIPAL = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/ex-diagnosistype", "code": "principal"}]}
GENDER = {"L": "male", "P": "female"}
SEVERITIES = ("0", "I", "II", "III")
SATUSEHAT_STAGING = "https://api-satusehat-stg.dto.kemkes.go.id"  # production: https://api-satusehat.kemkes.go.id


# ---------------------------------------------------------------- claims as FHIR

def claim_bundle(visits: pd.DataFrame) -> dict:
    """One patient's claims as a FHIR Bundle: Patient, then Organization, Encounter and Claim per claim.

    This is the shape `claims_from_bundle` reads back. The index of `visits` becomes the claim
    identifier. The sample data has an age per claim but no birth date, so birthDate is a date that
    gives every claim the age it has in the data.
    """
    newest = visits.iloc[-1]
    patient = {"resourceType": "Patient", "id": f"p{newest.patient}",
               "identifier": [{"system": BASE + "peserta", "value": str(newest.patient)}]}
    if newest.sex in GENDER:
        patient["gender"] = GENDER[newest.sex]
    aged = visits[visits.age >= 0]
    if len(aged):
        # age = days since birth // 365.25, so each claim allows a window of birth days; take the middle of the overlap
        day = (aged.admit - pd.Timestamp(0)).dt.days
        last, first = (day - np.ceil(aged.age * 365.25)).min(), (day - (aged.age + 1) * 365.25).max()
        patient["birthDate"] = f"{pd.Timestamp(0) + pd.Timedelta(days=int(last - max(0, (last - first) // 2))):%Y-%m-%d}"
    resources = {f"Patient/{patient['id']}": patient}
    for i, c in enumerate(visits.itertuples(), 1):
        provider = f"Organization/rs{c.hospital}-k{c.hosp_class}"  # the class can differ between claims
        resources.setdefault(provider, {
            "resourceType": "Organization", "id": provider.split("/")[1],
            "identifier": [{"system": BASE + "faskes", "value": str(c.hospital)}],
            "type": [{"coding": [{"system": HOSPITAL_CLASS, "code": str(c.hosp_class)}]}],
        })
        period = {"start": f"{c.admit:%Y-%m-%d}", "end": f"{c.discharge:%Y-%m-%d}"}
        resources[f"Encounter/e{i}"] = {
            "resourceType": "Encounter", "id": f"e{i}", "status": "finished",
            "identifier": [{"system": BASE + "kunjungan", "value": str(c.Index)}],
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                      "code": "IMP" if c.service == "RI" else "AMB"},
            "subject": {"reference": f"Patient/{patient['id']}"}, "period": period,
            "serviceProvider": {"reference": provider},
        }
        package = {"coding": [{"system": INA_CBG, "code": f"{c.cbg}-{c.severity}"}]}
        codes = [c.dx, *(code for code in (c.sec1, c.sec2, c.sec3) if isinstance(code, str))]
        resources[f"Claim/c{i}"] = {
            "resourceType": "Claim", "id": f"c{i}", "status": "active",
            "identifier": [{"system": BASE + "klaim", "value": str(c.Index)}],
            "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/claim-type", "code": "institutional"}]},
            "use": "claim", "patient": {"reference": f"Patient/{patient['id']}"},
            "billablePeriod": period, "created": period["end"],
            "provider": {"reference": provider},
            "priority": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/processpriority", "code": "normal"}]},
            "insurance": [{"sequence": 1, "focal": True, "coverage": {"display": "JKN"}}],
            "diagnosis": [
                {"sequence": n, "diagnosisCodeableConcept": {"coding": [{"system": ICD10, "code": code}]},
                 **({"type": [PRINCIPAL], "packageCode": package} if n == 1 else {})}
                for n, code in enumerate(codes, 1)
            ],
            "item": [{"sequence": 1, "encounter": [{"reference": f"Encounter/e{i}"}], "productOrService": package}],
            "total": {"value": int(c.cost), "currency": "IDR"},
        }
    return {"resourceType": "Bundle", "type": "collection",
            "entry": [{"fullUrl": BASE + ref, "resource": resource} for ref, resource in resources.items()]}


def _day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize(None) if stamp.tzinfo else stamp).normalize()


def _code(concept: dict) -> str:
    """The ICD-10 code of a CodeableConcept, cut to the 3 characters the model knows."""
    codings = concept["coding"]
    coding = next((c for c in codings if c.get("system") == ICD10), codings[0])
    return str(coding["code"]).upper().replace(".", "")[:3]


def _claim_row(claim: dict, found: dict, sex, birth) -> dict:
    admit, discharge = _day(claim["billablePeriod"]["start"]), _day(claim["billablePeriod"]["end"])
    diagnoses = sorted(claim["diagnosis"], key=lambda d: d.get("sequence", 0))
    principal = lambda d: any(c.get("code") == "principal" for t in d.get("type", []) for c in t.get("coding", []))
    main = next((d for d in diagnoses if principal(d)), diagnoses[0])
    concept = lambda d: d.get("diagnosisCodeableConcept") or found[d["diagnosisReference"]["reference"]]["code"]
    secondary = sorted({_code(concept(d)) for d in diagnoses if d is not main})[:3]
    item = (claim.get("item") or [{}])[0]
    package = (main.get("packageCode") or item["productOrService"])["coding"][0]["code"]
    cbg, _, severity = str(package).rpartition("-")
    provider = claim["provider"].get("reference", "")
    organization = found.get(provider, {})
    encounter = found.get((item.get("encounter") or [{}])[0].get("reference"), {})
    inpatient = encounter["class"]["code"] == "IMP" if encounter else severity != "0"
    classes = [c["code"] for t in organization.get("type", []) for c in t.get("coding", []) if c.get("system") == HOSPITAL_CLASS]
    return {
        "patient": 0, "admit": admit, "discharge": discharge, "los": (discharge - admit).days,
        "hospital": str((organization.get("identifier") or [{"value": provider}])[0]["value"]),
        "hosp_class": str(classes[0]) if classes else "?",
        "service": "RI" if inpatient else "RJ", "dx": _code(concept(main)), "cbg": cbg, "severity": severity,
        "sex": sex, "age": int((admit - birth).days // 365.25) if birth is not None else -1,
        **{f"sec{i + 1}": secondary[i] if i < len(secondary) else None for i in range(3)},
    }


def claims_from_bundle(bundle: dict):
    """A FHIR Bundle with one patient's Claims -> (rows shaped like data/claims.parquet, the newest Claim).

    The newest Claim is the one to assess; older ones are its history. Raises ValueError, with a
    message meant for the sender, when the Bundle cannot be read.
    """
    try:
        if bundle.get("resourceType") != "Bundle":
            raise ValueError("expected a FHIR Bundle")
        found = {}  # resources by "Type/id" and by fullUrl, the two ways a reference can point at them
        for entry in bundle["entry"]:
            resource = entry["resource"]
            found[f"{resource['resourceType']}/{resource.get('id')}"] = found[entry.get("fullUrl")] = resource
        claims = [entry["resource"] for entry in bundle["entry"] if entry["resource"]["resourceType"] == "Claim"]
        if not 1 <= len(claims) <= nadi.MAX_VISITS:
            raise ValueError(f"expected 1 to {nadi.MAX_VISITS} Claim resources, got {len(claims)}")
        if len({json.dumps(c["patient"], sort_keys=True) for c in claims}) != 1:
            raise ValueError("all Claims must be for the same patient")
        patient = found.get(claims[0]["patient"].get("reference"), {})
        birth = _day(patient["birthDate"]) if patient.get("birthDate") else None
        sex = {v: k for k, v in GENDER.items()}.get(patient.get("gender"))
        rows = [(_claim_row(c, found, sex, birth), c) for c in claims]
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"Bundle is missing something a Claim needs: {error!r}") from error
    for row, _ in rows:
        if row["severity"] not in SEVERITIES or not row["cbg"]:
            raise ValueError("packageCode must be an INA-CBG code ending in -0, -I, -II or -III")
        if not 0 <= row["los"] <= 3650:
            raise ValueError("billablePeriod.end must be on or after billablePeriod.start")
        if any(len(str(row[field])) > 16 for field in ("hospital", "hosp_class", "dx", "cbg")):
            raise ValueError("a code or identifier is longer than 16 characters")
    rows.sort(key=lambda pair: (pair[0]["admit"], pair[0]["discharge"]))
    return pd.DataFrame([row for row, _ in rows]), rows[-1][1]


# ------------------------------------------------------------ assessment as FHIR

def assessment_bundle(key: str, subject: dict, claim_ref: dict, result: dict, explanation: list,
                      reviews: list, version: str) -> dict:
    """The assessment as a FHIR R4 Bundle: RiskAssessment, plus who produced and who verified it."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    risk = {
        "resourceType": "RiskAssessment", "id": f"nadi-{key}",
        "meta": {"security": [AIAST]},
        "status": "final",
        "code": {"text": "Prioritas review klaim rawat inap JKN"},
        "method": {"text": f"NADI {version}: probabilitas tiap field klaim, diprediksi dari riwayat klaim peserta"},
        "subject": subject,
        "occurrenceDateTime": now,
        "performer": {"reference": "Device/nadi"},
        "basis": [claim_ref],
        "prediction": [
            {"outcome": {"text": f"Severity level {s['level']}"}, "probabilityDecimal": round(s["p"], 4),
             **({"rationale": "Severity yang diklaim"} if s["claimed"] else {})}
            for s in result["severity"]
        ],
        "note": [{"text": " ".join([*explanation, "Ini prioritas review, bukan keputusan. Keputusan tetap di verifikator."])},
                 *({"time": r["time"], "authorString": r["reviewer"],
                    "text": f"Keputusan verifikator: {r['decision']}. {r['note']}".strip()} for r in reviews)],
    }
    # the reviewer who made the latest decision, if any
    verifier = [{"resourceType": "Practitioner", "id": f"verifikator-{key}", "name": [{"text": r["reviewer"]}]}
                for r in reviews[-1:]]
    agent_type = lambda code: {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/provenance-participant-type", "code": code}]}
    provenance = {
        "resourceType": "Provenance", "id": f"nadi-{key}",
        "meta": {"profile": [AI_IG + "/StructureDefinition/AI-Provenance"]},
        "target": [{"reference": f"RiskAssessment/nadi-{key}"}],
        "recorded": now,
        "reason": [{"coding": [AIAST]}],
        "agent": [{"type": agent_type("author"), "role": [AI_TYPE], "who": {"reference": "Device/nadi"}},
                  *({"type": agent_type("verifier"), "who": {"reference": f"Practitioner/{v['id']}"}} for v in verifier)],
        "entity": [{"role": "source", "what": claim_ref}],
    }
    device = {
        "resourceType": "Device", "id": "nadi",
        "meta": {"profile": [AI_IG + "/StructureDefinition/AI-Device"]},
        "identifier": [{"system": BASE + "model", "value": version}],
        "type": AI_TYPE,
        "deviceName": [{"name": "NADI", "type": "model-name"}],
        "version": [{"value": version}],
        "manufacturer": "SmartGuard JKN",
    }
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"fullUrl": f"{BASE}{r['resourceType']}/{r['id']}", "resource": r}
        for r in (risk, provenance, device, *verifier)]}


def for_satusehat(risk: dict) -> dict:
    """The RiskAssessment in the shape SATUSEHAT documents: the server assigns the id, and no Device.

    SATUSEHAT has no Device or Provenance resource, so the model identity travels in `method` and
    in the AIAST label, and the full Provenance stays on our side.
    """
    return {k: v for k, v in risk.items() if k not in ("id", "performer")}


def _relink(node, urn: dict) -> None:
    if isinstance(node, dict):
        if node.get("reference") in urn:
            node["reference"] = urn[node["reference"]]
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            _relink(child, urn)


def as_transaction(*bundles: dict) -> dict:
    """The resources of `bundles` as one transaction: the server stores them together and links them itself.

    A resource that carries an identifier is created only if the server does not have it yet, so
    sending the same claim again adds a new assessment without duplicating patient, facility or claim.
    """
    entries, urn = [], {}
    for entry in copy.deepcopy([entry for bundle in bundles for entry in bundle["entry"]]):
        resource, uid = entry["resource"], f"urn:uuid:{uuid.uuid4()}"
        urn[f"{resource['resourceType']}/{resource.pop('id', None)}"] = urn[entry.get("fullUrl")] = uid
        request = {"method": "POST", "url": resource["resourceType"]}
        for identifier in (resource.get("identifier") or [])[:1]:
            token = urllib.parse.quote(f"{identifier.get('system', '')}|{identifier['value']}", safe="")
            request["ifNoneExist"] = "identifier=" + token
        entries.append({"fullUrl": uid, "resource": resource, "request": request})
    _relink(entries, urn)
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


# ----------------------------------------------------------------------- client

_token = {"value": "", "expires": 0.0}
_token_lock = threading.Lock()


def target() -> dict:
    """Where send() posts to. Decided by the environment only, never by a request."""
    env = os.environ
    if env.get("SATUSEHAT_CLIENT_ID") and env.get("SATUSEHAT_CLIENT_SECRET"):
        host = env.get("SATUSEHAT_HOST", SATUSEHAT_STAGING).rstrip("/")
        where = {"mode": "satusehat", "base": host + "/fhir-r4/v1", "auth": host + "/oauth2/v1"}
    elif env.get("FHIR_BASE"):
        where = {"mode": "fhir", "base": env["FHIR_BASE"].rstrip("/")}
    else:
        where = {"mode": "dry-run", "base": SATUSEHAT_STAGING + "/fhir-r4/v1", "auth": SATUSEHAT_STAGING + "/oauth2/v1"}
    if not all(url.startswith(("https://", "http://")) for url in (where["base"], where.get("auth", "https://"))):
        raise ValueError("the FHIR server address must start with http:// or https://")
    return where


def _post(url: str, data: bytes, headers: dict):
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:  # FHIR servers explain a refusal in an OperationOutcome body
        status, body = error.code, error.read()
    try:
        return status, json.loads(body or b"{}")
    except ValueError:
        return status, {"text": body.decode(errors="replace")[:500]}


def _bearer(auth: str) -> str:
    """SATUSEHAT access token (OAuth2 client credentials), reused until shortly before it expires."""
    with _token_lock:
        if time.time() >= _token["expires"]:
            form = urllib.parse.urlencode({"client_id": os.environ["SATUSEHAT_CLIENT_ID"],
                                           "client_secret": os.environ["SATUSEHAT_CLIENT_SECRET"]})
            status, body = _post(auth + "/accesstoken?grant_type=client_credentials", form.encode(),
                                 {"Content-Type": "application/x-www-form-urlencoded"})
            if status != 200 or "access_token" not in body:
                raise ConnectionError(f"SATUSEHAT token request failed with HTTP {status}")
            _token.update(value=body["access_token"], expires=time.time() + int(body.get("expires_in", 3599)) - 60)
        return _token["value"]


def describe(resource: dict) -> dict:
    """The requests a SATUSEHAT delivery consists of, without sending anything and without any secret."""
    where = target()
    auth = where.get("auth", SATUSEHAT_STAGING + "/oauth2/v1")
    base = where["base"] if "auth" in where else SATUSEHAT_STAGING + "/fhir-r4/v1"
    return {
        "sent": False,
        "token": {"method": "POST", "url": auth + "/accesstoken?grant_type=client_credentials",
                  "headers": {"Content-Type": "application/x-www-form-urlencoded"},
                  "body": "client_id=<client_id>&client_secret=<client_secret>"},
        "request": {"method": "POST", "url": f"{base}/{resource['resourceType']}",
                    "headers": {"Authorization": "Bearer <access_token>", "Content-Type": "application/json"},
                    "body": resource},
    }


def send(resource: dict) -> dict:
    """POST one resource, or a transaction Bundle, to the configured server. A dry run only describes it."""
    where = target()
    if where["mode"] == "dry-run":
        return describe(resource)
    url = where["base"] + ("" if resource.get("type") == "transaction" else "/" + resource["resourceType"])
    headers = {"Content-Type": "application/json"}
    if where["mode"] == "satusehat":
        headers["Authorization"] = "Bearer " + _bearer(where["auth"])
    status, body = _post(url, json.dumps(resource).encode(), headers)
    return {"sent": True, "mode": where["mode"], "url": url, "status": status, "response": body}
