"""Quick check for the tokenizer, the rule-based flags and the FHIR bridge. Run: python test_nadi.py"""
import copy
import http.server
import json
import os
import threading
from pathlib import Path

import pandas as pd

import bridge
import nadi


def demo() -> None:
    day = pd.Timestamp("2024-01-01")
    after = lambda n: day + pd.Timedelta(days=n)
    df = pd.DataFrame({
        "patient": [0, 0, 0, 1],
        "admit": [day, after(5), after(40), day],
        "discharge": [after(4), after(8), after(40), day],
        "hospital": [7, 7, 9, 7], "hosp_class": [3, 3, 1, 3],
        "service": ["RI", "RI", "RJ", "RJ"], "dx": ["A09", "A09", "Z09", "J06"],
        "cbg": ["K-4-17", "K-4-17", "Q-5-44", "Q-5-44"], "severity": ["I", "II", "0", "0"],
        "los": [4, 3, 0, 0], "sex": ["L", "L", "L", "P"], "age": [30, 30, 30, -1],
        "sec1": [None, "E86", None, None], "sec2": None, "sec3": None,
    })

    feats = nadi.features(df)
    assert feats.gap.tolist() == ["first", "1", "31-90", "first"]
    assert feats.fac.tolist() == ["first", "same", "diff", "first"]
    assert feats.los.tolist() == ["4-5", "3", "0", "0"]
    assert feats.demo.tolist() == ["L:18-39", "L:18-39", "L:18-39", "P:?"]
    assert feats.sec1.tolist() == ["-", "E86", "-", "-"]

    # readmitted a day after discharge, same hospital and dx
    assert nadi.rule_flags(df).split_episode.tolist() == [False, True, False, False]

    vocab = nadi.build_vocab(feats)  # with data this small every value falls back to "?"
    tokens = nadi.encode(feats, vocab)
    assert (tokens[:, nadi.FIELDS.index("dx")] == vocab["dx=?"]).all()
    X, chunk, pos = nadi.to_sequences(tokens, df.patient.to_numpy())
    assert X.shape == (2, nadi.MAX_VISITS, len(nadi.FIELDS))
    assert chunk.tolist() == [0, 0, 0, 1] and pos.tolist() == [0, 1, 2, 0]
    assert (X[1, 1:] == nadi.PAD).all()

    # facility 1 codes severity II three times as often as its peers
    n = nadi.MIN_FACILITY_CLAIMS
    stays = pd.DataFrame({
        "hospital": [h for h in range(1, 11) for _ in range(n)],
        "cbg": "K-4-17", "hosp_class": 3,
        "severity": ["II"] * 18 + ["I"] * 12 + (["II"] * 6 + ["I"] * 24) * 9,
    })
    index = nadi.facility_index(stays, nadi.peer_expected(stays))
    assert index.observed[1] == 18 and abs(index.expected[1] - 7.2) < 1e-9
    assert index.z.idxmax() == 1 and index.z[1] > 3 > index.z[2]
    assert nadi.facility_index(stays.iloc[1:], nadi.peer_expected(stays).iloc[1:]).index.tolist() == list(range(2, 11))

    fhir_checks(df[df.patient == 0].assign(cost=1000))
    client_checks()
    if Path("artifacts/scored.parquet").exists():
        app_checks()
    else:
        print("skipped the app checks: no trained model and scored claims here")
    print("ok")


def refused(bundle) -> bool:
    try:
        bridge.claims_from_bundle(bundle)
    except ValueError:
        return True
    return False


def fhir_checks(visits: pd.DataFrame) -> None:
    """Claims survive the trip to FHIR and back with every model input intact."""
    bundle = json.loads(json.dumps(bridge.claim_bundle(visits)))
    back, newest = bridge.claims_from_bundle(bundle)
    assert newest["id"] == "c3"
    assert (nadi.features(back).to_numpy() == nadi.features(visits).to_numpy()).all()

    def changed(edit) -> dict:
        other = copy.deepcopy(bundle)
        edit([entry["resource"] for entry in other["entry"] if entry["resource"]["resourceType"] == "Claim"])
        return other

    assert refused({}) and refused({"resourceType": "Bundle", "entry": []}) and refused("Bundle")
    assert refused(changed(lambda claims: claims[0]["patient"].update(reference="Patient/someone-else")))
    assert refused(changed(lambda claims: claims[0]["diagnosis"][0]["packageCode"]["coding"][0].update(code="K-4-17-IV")))
    assert refused(changed(lambda claims: claims[0]["billablePeriod"].update(end="2023-01-01")))
    assert refused(changed(lambda claims: claims[0].pop("diagnosis")))

    # as a transaction every reference points at an entry of the same Bundle
    transaction = bridge.as_transaction(bundle)
    urns = {entry["fullUrl"] for entry in transaction["entry"]}
    references = [part.split('"')[0] for part in json.dumps(transaction).split('"reference": "')[1:]]
    assert references and set(references) <= urns and all("id" not in e["resource"] for e in transaction["entry"])


class FakeSatusehat(http.server.BaseHTTPRequestHandler):
    """Answers like the documented SATUSEHAT endpoints: one token route, one RiskAssessment route."""
    calls = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        self.calls.append(self.path)
        if self.path == "/oauth2/v1/accesstoken?grant_type=client_credentials":
            form = self.headers["Content-Type"] == "application/x-www-form-urlencoded"
            ok = form and body == "client_id=my-id&client_secret=my-secret"
            status, reply = (200, {"access_token": "t0k3n", "expires_in": "3599", "token_type": "BearerToken"}) if ok else (401, {})
        elif self.path != "/fhir-r4/v1/RiskAssessment" or self.headers["Authorization"] != "Bearer t0k3n":
            status, reply = 401, {}
        elif json.loads(body)["status"] == "final":
            status, reply = 201, {**json.loads(body), "id": "abc"}
        else:
            status, reply = 400, {"resourceType": "OperationOutcome", "issue": [{"severity": "error", "code": "invalid"}]}
        payload = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def client_checks() -> None:
    """The client follows the SATUSEHAT token and POST contract, and a dry run sends nothing."""
    risk = {"resourceType": "RiskAssessment", "status": "final"}
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeSatusehat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    settings = {"SATUSEHAT_CLIENT_ID": "my-id", "SATUSEHAT_CLIENT_SECRET": "my-secret",
                "SATUSEHAT_HOST": f"http://127.0.0.1:{server.server_port}"}
    os.environ.update(settings)
    try:
        accepted, rejected = bridge.send(risk), bridge.send({**risk, "status": "draft"})
    finally:
        for name in settings:
            del os.environ[name]
        server.shutdown()
    assert accepted["status"] == 201 and accepted["response"]["id"] == "abc"
    assert rejected["status"] == 400 and rejected["response"]["resourceType"] == "OperationOutcome"
    # one token request serves both deliveries
    assert FakeSatusehat.calls == ["/oauth2/v1/accesstoken?grant_type=client_credentials"] + ["/fhir-r4/v1/RiskAssessment"] * 2

    if not os.environ.get("FHIR_BASE"):
        preview = bridge.send(risk)
        assert preview["sent"] is False and len(FakeSatusehat.calls) == 3
        assert preview["request"]["url"] == bridge.SATUSEHAT_STAGING + "/fhir-r4/v1/RiskAssessment"
        assert "my-secret" not in json.dumps(preview)


def app_checks() -> None:
    """With the real model: the explanation follows the numbers, and a claim scores the same through FHIR."""
    import app

    few = app.claims.groupby("patient").patient.transform("size") <= nadi.MAX_VISITS  # history fits one sequence
    for row in app.claims[(app.claims.potential > 0) & few].index[:5]:
        stored = app.assess(row, {})
        assert stored["explanation"][0].startswith("RS menagih severity")
        assert "lebih rendah" in stored["explanation"][1]
        as_expected = app.assess(row, {"sev": app.claims.expected[row]})
        assert "sesuai perkiraan NADI" in as_expected["explanation"][0]
        assert not any("lebih rendah" in sentence for sentence in as_expected["explanation"])

        visits, _ = bridge.claims_from_bundle(app.fhir_claim(row))
        through_fhir = app.score(nadi.features(visits), len(visits) - 1)
        for a, b in zip(stored["severity"], through_fhir["severity"]):
            assert a["claimed"] == b["claimed"] and abs(a["p"] - b["p"]) < 1e-6


if __name__ == "__main__":
    demo()
