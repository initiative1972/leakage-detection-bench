"""The authored benchmark items (the human-editable source of truth).

``scripts/build_dataset.py`` validates these and serialises them to the
distributable ``data/benchmark.jsonl``. Keeping the source here (typed, in one
place) makes authoring, review and schema-validation straightforward.

All content is synthetic and illustrative. No private or employer data is used.
Every spec is a *plausible* ML task; the label records whether it contains a
leakage flaw and why. Descriptions are written the way a data scientist would
actually encounter them -- the leak must be *reasoned about*, not read off a
single machine-readable flag.
"""
from __future__ import annotations

from typing import List

from .schema import BenchItem, Feature, Label, TaskSpec


def _f(name: str, dtype: str, desc: str) -> Feature:
    return Feature(name=name, dtype=dtype, description=desc)


def mk(id, domain, task_type, target, prediction_time, split, preprocessing,
       features, has_leakage, leak_type, offending, rationale, difficulty,
       notes=""):
    spec = TaskSpec(
        id=id, domain=domain, task_type=task_type, target=target,
        prediction_time=prediction_time, split=split, preprocessing=preprocessing,
        features=features, notes=notes,
    )
    label = Label(
        has_leakage=has_leakage, leak_type=leak_type,
        offending_features=offending, rationale=rationale, difficulty=difficulty,
    )
    return BenchItem(spec=spec, label=label)


ITEMS: List[BenchItem] = [

    # ----------------------------------------------------------------- #
    # target_leakage -- a feature is a consequence / proxy of the label #
    # ----------------------------------------------------------------- #
    mk(
        "TL01", "Retail banking - credit risk", "classification",
        "customer defaults within the next 90 days",
        "scored on the 1st of each month using data available to the prior "
        "month end",
        "time-based: train 2019-2021, test 2022",
        "numeric features standardised within training folds only",
        [
            _f("months_on_book", "int", "account tenure in months at scoring"),
            _f("utilization_ratio", "float", "statement balance divided by limit"),
            _f("recovery_amount", "float",
               "dollars recovered by collections after the account was "
               "charged off for this default"),
            _f("collections_calls", "int",
               "number of collections calls placed about this default episode"),
        ],
        True, "target_leakage", ["recovery_amount", "collections_calls"],
        "Recoveries and collections activity only exist *because* the customer "
        "defaulted; both are recorded after the outcome and are unavailable at "
        "scoring time. They are direct consequences of the label.",
        "easy",
    ),
    mk(
        "TL02", "Healthcare - hospital operations", "classification",
        "patient is readmitted within 30 days of discharge",
        "scored at the moment of discharge",
        "random 80/20 split stratified by the label",
        "one-hot encoding fit on the training split",
        [
            _f("age", "int", "patient age at admission"),
            _f("primary_diagnosis", "cat", "ICD code for the index admission"),
            _f("length_of_stay", "float", "days of the index stay"),
            _f("readmit_drg", "cat",
               "DRG billing code of the readmission encounter"),
        ],
        True, "target_leakage", ["readmit_drg"],
        "readmit_drg is the billing code of the readmission itself; it can only "
        "be populated once a readmission has occurred, so it encodes the label.",
        "medium",
    ),
    mk(
        "TL03", "Payments - card fraud", "classification",
        "transaction is fraudulent",
        "scored in real time at authorisation",
        "time-ordered split by transaction timestamp",
        "amount log-transformed; encoders fit on train only",
        [
            _f("amount", "float", "transaction amount"),
            _f("merchant_category", "cat", "MCC of the merchant"),
            _f("distance_from_home", "float", "km from the cardholder's home"),
            _f("chargeback_filed", "bool",
               "whether the cardholder later filed a chargeback disputing this "
               "transaction"),
        ],
        True, "target_leakage", ["chargeback_filed"],
        "A chargeback is filed days to weeks after the transaction and is a "
        "downstream consequence of fraud; it is not known at authorisation.",
        "easy",
    ),
    mk(
        "TL04", "Telecom - subscriber churn", "classification",
        "subscriber cancels within 60 days",
        "scored monthly on the active subscriber base",
        "split by calendar month (train earlier months, test later)",
        "standardisation fit per training fold",
        [
            _f("tenure_months", "int", "months since activation"),
            _f("monthly_charge", "float", "current plan price"),
            _f("support_tickets_90d", "int", "tickets opened in the last 90 days"),
            _f("cancellation_reason", "cat",
               "reason code the retention agent records when a subscriber calls "
               "to cancel"),
        ],
        True, "target_leakage", ["cancellation_reason"],
        "cancellation_reason is captured only during a cancellation call, so it "
        "exists only for churners and directly reveals the outcome.",
        "medium",
    ),

    # ----------------------------------------------------------------- #
    # temporal_leakage -- feature uses post-prediction-time information #
    # ----------------------------------------------------------------- #
    mk(
        "TE01", "Retail - demand forecasting", "regression",
        "units sold of a SKU in the upcoming week",
        "forecast produced on the Friday before the target week",
        "time-based split by week",
        "lag features; scaling fit on train only",
        [
            _f("sales_lag_1w", "float", "units sold one week earlier"),
            _f("sales_lag_4w", "float", "units sold four weeks earlier"),
            _f("promo_flag", "bool",
               "whether a promotion actually ran during the target week; the "
               "promo calendar is finalised at month end, after the forecast"),
        ],
        True, "temporal_leakage", ["promo_flag"],
        "promo_flag describes the target week itself and is only finalised after "
        "the forecast is made; using it injects future information.",
        "medium",
    ),
    mk(
        "TE02", "Lending - portfolio risk", "classification",
        "loan defaults during its term",
        "scored at loan origination",
        "random split by loan id",
        "bureau features standardised on train only",
        [
            _f("loan_amount", "float", "principal at origination"),
            _f("dti_at_origination", "float", "debt-to-income at origination"),
            _f("bureau_score", "int",
               "credit-bureau score pulled at the quarterly portfolio review, "
               "six months after origination"),
        ],
        True, "temporal_leakage", ["bureau_score"],
        "The bureau score is a snapshot taken six months after the prediction "
        "point; at origination that future value is unknown.",
        "hard",
    ),
    mk(
        "TE03", "Healthcare - capacity planning", "regression",
        "length of stay for an admitted patient",
        "predicted at the hour of admission",
        "time-based split by admission date",
        "imputation fit within each training fold",
        [
            _f("age", "int", "patient age"),
            _f("admission_type", "cat", "elective / emergency / urgent"),
            _f("num_procedures", "int",
               "total number of procedures performed during the entire stay"),
        ],
        True, "temporal_leakage", ["num_procedures"],
        "The procedure count accrues across the whole stay and is only final at "
        "discharge; at admission it lies in the future.",
        "medium",
    ),
    mk(
        "TE04", "AdTech - click prediction", "classification",
        "user clicks the ad impression",
        "scored at the instant the impression is served",
        "time-ordered split",
        "hashing encoder fit on train only",
        [
            _f("ad_position", "int", "slot index on the page"),
            _f("user_ctr_7d", "float", "user click-through rate over prior 7 days"),
            _f("dwell_time_s", "float",
               "seconds the user spent on the landing page during this session"),
        ],
        True, "temporal_leakage", ["dwell_time_s"],
        "Landing-page dwell time happens only after a click and after the "
        "impression decision; it is unavailable at impression time.",
        "medium",
    ),

    # ----------------------------------------------------------------- #
    # preprocessing_leakage -- transform fit on all data before split   #
    # ----------------------------------------------------------------- #
    mk(
        "PP01", "Manufacturing - quality control", "classification",
        "unit fails final inspection",
        "scored at end of line",
        "random 70/30 split",
        "StandardScaler fit on the entire dataset before the train/test split, "
        "then the same scaler applied to both parts",
        [
            _f("torque", "float", "fastening torque"),
            _f("temperature", "float", "chamber temperature"),
            _f("cycle_time_s", "float", "assembly cycle time"),
        ],
        True, "preprocessing_leakage", [],
        "Fitting the scaler on all rows before splitting leaks test-set "
        "statistics (mean/variance) into training.",
        "easy",
    ),
    mk(
        "PP02", "Insurance - claims", "classification",
        "claim is fraudulent",
        "scored when the claim is filed",
        "5-fold cross-validation",
        "missing values imputed with the column mean computed over all rows "
        "(train and test together) before folds are created",
        [
            _f("claim_amount", "float", "amount claimed"),
            _f("policy_age_days", "int", "days since policy start"),
            _f("prior_claims", "int", "count of prior claims"),
        ],
        True, "preprocessing_leakage", [],
        "Global-mean imputation computed over all rows lets each fold's "
        "validation data influence the imputed values used in training.",
        "medium",
    ),
    mk(
        "PP03", "Credit - default", "classification",
        "borrower defaults within 12 months",
        "scored at application",
        "random split",
        "SMOTE oversampling applied to the full dataset to balance classes, "
        "then the balanced data is split into train and test",
        [
            _f("income", "float", "stated annual income"),
            _f("dti", "float", "debt-to-income ratio"),
            _f("num_open_lines", "int", "open credit lines"),
        ],
        True, "preprocessing_leakage", [],
        "Oversampling before the split copies/synthesises minority points that "
        "then appear (or inform neighbours) in both train and test.",
        "medium",
    ),
    mk(
        "PP04", "NLP - support ticket routing", "classification",
        "ticket is high priority",
        "scored at ticket creation",
        "random split",
        "the top 200 TF-IDF terms are selected by mutual information with the "
        "label computed on all rows, then cross-validation is run on the "
        "reduced feature set",
        [
            _f("subject_text", "text", "ticket subject line"),
            _f("body_text", "text", "ticket body"),
            _f("channel", "cat", "email / chat / phone"),
        ],
        True, "preprocessing_leakage", [],
        "Supervised feature selection using the label over all rows (including "
        "test) is a classic leak; selection must happen inside each fold.",
        "hard",
    ),

    # ----------------------------------------------------------------- #
    # group_leakage -- same entity in train and test (row-level split)  #
    # ----------------------------------------------------------------- #
    mk(
        "GR01", "Healthcare - diagnosis", "classification",
        "an imaging study shows disease",
        "scored per study",
        "random 80/20 split over individual images",
        "augmentation applied within training only",
        [
            _f("image_embedding", "vector", "CNN embedding of the scan"),
            _f("modality", "cat", "CT / MRI / X-ray"),
            _f("patient_id", "id", "identifier of the patient"),
        ],
        True, "group_leakage", [],
        "Patients have multiple images; a random per-image split puts different "
        "images of the same patient in both train and test, inflating scores. "
        "The split should be grouped by patient_id.",
        "medium",
        notes="Each patient contributes several studies over time.",
    ),
    mk(
        "GR02", "E-commerce - review moderation", "classification",
        "a product review is abusive",
        "scored per review",
        "random split over reviews",
        "text vectoriser fit on train only",
        [
            _f("review_text", "text", "the review body"),
            _f("stars", "int", "star rating"),
            _f("reviewer_id", "id", "identifier of the reviewer"),
        ],
        True, "group_leakage", [],
        "Prolific reviewers write many reviews in a recognisable style; random "
        "splitting leaks a reviewer's style across train and test. Split by "
        "reviewer_id.",
        "hard",
        notes="A minority of reviewers account for most reviews.",
    ),
    mk(
        "GR03", "IoT - predictive maintenance", "classification",
        "a machine fails within 24 hours",
        "scored hourly per machine",
        "random split over hourly records",
        "scaling fit on train only",
        [
            _f("vibration_rms", "float", "vibration RMS this hour"),
            _f("temp_c", "float", "bearing temperature this hour"),
            _f("machine_id", "id", "identifier of the machine"),
        ],
        True, "group_leakage", [],
        "Consecutive hourly rows from one machine are highly correlated; a "
        "random split leaks a machine's regime across train and test. Use a "
        "machine-grouped (or time-based) split.",
        "medium",
        notes="Thousands of hourly rows per machine.",
    ),
    mk(
        "GR04", "Banking - application scoring", "classification",
        "a credit-card application is approved",
        "scored per application",
        "random split over applications",
        "encoders fit on train only",
        [
            _f("requested_limit", "float", "requested credit limit"),
            _f("income", "float", "stated income"),
            _f("applicant_id", "id", "identifier of the applicant"),
        ],
        True, "group_leakage", [],
        "Repeat applicants submit multiple applications with near-identical "
        "attributes; random splitting leaks the same applicant across folds. "
        "Group by applicant_id.",
        "hard",
        notes="Many applicants re-apply several times.",
    ),

    # ----------------------------------------------------------------- #
    # proxy_leakage -- an id / ordering / code that encodes the label   #
    # ----------------------------------------------------------------- #
    mk(
        "PX01", "Fraud - labelled dump", "classification",
        "account is fraudulent",
        "scored per account",
        "random split",
        "numeric scaling on train only",
        [
            _f("balance", "float", "account balance"),
            _f("num_logins_30d", "int", "logins in last 30 days"),
            _f("record_id", "int",
               "auto-increment row id; confirmed-fraud accounts were bulk-loaded "
               "into the table last, so larger ids skew fraudulent"),
        ],
        True, "proxy_leakage", ["record_id"],
        "record_id is an artefact of load order that correlates with the label; "
        "it carries no causal signal and must be dropped.",
        "medium",
    ),
    mk(
        "PX02", "Healthcare - multi-site study", "classification",
        "tumour is malignant",
        "scored per case",
        "random split",
        "features standardised on train only",
        [
            _f("lesion_size_mm", "float", "largest lesion diameter"),
            _f("age", "int", "patient age"),
            _f("mrn", "id",
               "medical record number; a specialist cancer centre with a high "
               "malignancy rate was assigned a contiguous MRN block"),
        ],
        True, "proxy_leakage", ["mrn"],
        "The MRN range encodes the site, and site is correlated with the "
        "outcome; the model can exploit the id instead of clinical signal.",
        "hard",
    ),
    mk(
        "PX03", "ML ops - dataset ordering", "classification",
        "email is spam",
        "scored per email",
        "the first 80% of rows are used for training and the last 20% for test",
        "no shuffling is applied before the split",
        [
            _f("row_index", "int", "position of the row in the delivered file"),
            _f("num_links", "int", "count of hyperlinks"),
            _f("subject_len", "int", "subject length in characters"),
        ],
        True, "proxy_leakage", ["row_index"],
        "The file is sorted with all spam first; an unshuffled positional split "
        "makes row_index (and the split itself) a proxy for the label.",
        "medium",
    ),
    mk(
        "PX04", "Vendor data - merged sources", "classification",
        "lead converts to a sale",
        "scored per lead",
        "random split",
        "one-hot on train only",
        [
            _f("lead_score", "float", "third-party propensity score"),
            _f("region", "cat", "sales region"),
            _f("source_file", "cat",
               "which vendor file the row came from; vendor A supplied only "
               "converted leads and vendor B only non-converters"),
        ],
        True, "proxy_leakage", ["source_file"],
        "source_file perfectly separates the classes by construction of the "
        "merge; it is a label proxy, not a usable feature.",
        "easy",
    ),

    # ----------------------------------------------------------------- #
    # none -- clean, well-posed specs (some contain benign cue words)   #
    # ----------------------------------------------------------------- #
    mk(
        "CL01", "Retail banking - credit risk", "classification",
        "customer defaults within the next 90 days",
        "scored at month end using only prior data",
        "time-based split, train earlier than test",
        "standardisation fit within each training fold only",
        [
            _f("months_on_book", "int", "tenure at scoring"),
            _f("utilization_ratio", "float", "balance over limit at month end"),
            _f("last_payment_date", "date",
               "date of the last payment made before the scoring date"),
        ],
        False, "none", [],
        "All features are known at scoring time; last_payment_date is a past "
        "event and the scaler is fit per fold. No leakage.",
        "medium",
    ),
    mk(
        "CL02", "Telecom - churn", "classification",
        "subscriber cancels within 60 days",
        "scored monthly",
        "split by customer so no customer appears in both train and test",
        "encoders fit on train only",
        [
            _f("tenure_months", "int", "months since activation"),
            _f("avg_monthly_usage_gb", "float", "mean data use over prior 3 months"),
            _f("support_tickets_90d", "int", "tickets in last 90 days"),
        ],
        False, "none", [],
        "Features are all pre-scoring and the split is grouped by customer, "
        "avoiding group leakage.",
        "easy",
    ),
    mk(
        "CL03", "Retail - demand forecasting", "regression",
        "units sold next week",
        "forecast on the Friday before the target week",
        "time-based split by week",
        "scaling fit on train only",
        [
            _f("sales_lag_1w", "float", "units sold one week earlier"),
            _f("sales_lag_4w", "float", "units sold four weeks earlier"),
            _f("planned_promo_flag", "bool",
               "whether a promotion was *scheduled and locked* before the "
               "forecast date for the target week"),
        ],
        False, "none", [],
        "The promo flag here is the plan locked before the forecast date, so it "
        "is legitimately known at prediction time (contrast with TE01).",
        "hard",
    ),
    mk(
        "CL04", "Payments - fraud", "classification",
        "transaction is fraudulent",
        "scored at authorisation",
        "time-ordered split",
        "log transform; encoders fit on train only",
        [
            _f("amount", "float", "transaction amount"),
            _f("merchant_category", "cat", "merchant MCC"),
            _f("account_age_days", "int", "days since the account was opened"),
        ],
        False, "none", [],
        "Every feature is available at authorisation time; account_age_days is a "
        "past-dated duration, not future information.",
        "medium",
    ),
    mk(
        "CL05", "Healthcare - readmission", "classification",
        "patient readmitted within 30 days",
        "scored at discharge",
        "split grouped by patient id",
        "imputation fit within folds",
        [
            _f("age", "int", "age at admission"),
            _f("num_prior_admissions", "int", "admissions in the prior year"),
            _f("discharge_disposition", "cat",
               "where the patient was discharged to (home, SNF, ...)"),
        ],
        False, "none", [],
        "discharge_disposition is decided at discharge (the prediction time) and "
        "the split is patient-grouped. No leak.",
        "hard",
    ),
    mk(
        "CL06", "Insurance - pricing", "regression",
        "annual claim cost for a policy",
        "scored when a quote is produced",
        "random split by policy",
        "scaling fit on train only",
        [
            _f("vehicle_age", "int", "age of the insured vehicle"),
            _f("driver_age", "int", "age of the primary driver"),
            _f("annual_mileage", "float", "declared annual mileage"),
        ],
        False, "none", [],
        "All inputs are declared at quote time; policies are independent so a "
        "random split is fine.",
        "easy",
    ),
    mk(
        "CL07", "IoT - predictive maintenance", "classification",
        "machine fails within 24 hours",
        "scored hourly",
        "split grouped by machine and ordered in time",
        "scaling fit on train only",
        [
            _f("vibration_rms_1h", "float", "vibration RMS this hour"),
            _f("temp_trend_6h", "float", "temperature slope over prior 6 hours"),
            _f("hours_since_service", "float", "hours since last maintenance"),
        ],
        False, "none", [],
        "Features use only information up to the prediction hour and the split "
        "is grouped by machine; no group or temporal leak.",
        "medium",
    ),
    mk(
        "CL08", "Marketing - response", "classification",
        "customer responds to a campaign",
        "scored before send",
        "random holdout",
        "encoders fit on train only",
        [
            _f("recency_days", "int", "days since last purchase"),
            _f("frequency_12m", "int", "purchases in last 12 months"),
            _f("campaign_id", "id",
               "random UUID assigned to each send, independent of the outcome"),
        ],
        False, "none", [],
        "campaign_id is a random identifier uncorrelated with the label; the "
        "RFM features are pre-send. The id mention is a benign distractor.",
        "medium",
    ),
    mk(
        "CL09", "Real estate - valuation", "regression",
        "sale price of a property",
        "scored at listing",
        "time-based split by listing date",
        "scaling fit on train only",
        [
            _f("living_area_sqm", "float", "interior area"),
            _f("bedrooms", "int", "number of bedrooms"),
            _f("year_built", "int", "construction year"),
        ],
        False, "none", [],
        "Property attributes are all known at listing; the date is used only to "
        "form the split, not as a feature.",
        "easy",
    ),
    mk(
        "CL10", "AdTech - click prediction", "classification",
        "user clicks the impression",
        "scored at impression",
        "time-ordered split",
        "hashing encoder fit on train only",
        [
            _f("ad_position", "int", "slot index"),
            _f("user_ctr_7d", "float", "user CTR over prior 7 days"),
            _f("hour_of_day", "int", "local hour at impression"),
        ],
        False, "none", [],
        "All features are available when the impression is served; nothing uses "
        "post-click information.",
        "medium",
    ),
    mk(
        "CL11", "HR - attrition", "classification",
        "employee leaves within 6 months of the review",
        "scored on the annual review date",
        "time-based split by review year",
        "standardisation fit on train only",
        [
            _f("tenure_years", "float", "years at company at review"),
            _f("last_rating", "cat", "most recent performance rating"),
            _f("commute_km", "float", "home-to-office distance"),
        ],
        False, "none", [],
        "The review-date snapshot precedes the 6-month outcome window; all "
        "features are known at review time.",
        "medium",
    ),
    mk(
        "CL12", "E-commerce - conversion", "classification",
        "session ends in a purchase",
        "scored at the add-to-cart event",
        "split grouped by user",
        "scaling fit on train only",
        [
            _f("items_in_cart", "int", "items in cart at the decision point"),
            _f("session_clicks_so_far", "int", "clicks before add-to-cart"),
            _f("is_returning_user", "bool", "whether the user has purchased before"),
        ],
        False, "none", [],
        "Features are measured up to the add-to-cart decision and the split is "
        "user-grouped; nothing uses the post-decision session tail.",
        "hard",
    ),
    mk(
        "CL13", "Healthcare - early warning", "classification",
        "patient develops sepsis in the next 6 hours",
        "scored each hour using vitals up to that hour",
        "time-based split, grouped by patient",
        "imputation fit within folds",
        [
            _f("heart_rate_last", "float", "most recent heart rate"),
            _f("lactate_last", "float", "most recent lactate"),
            _f("hours_in_icu", "float", "hours in ICU so far"),
        ],
        False, "none", [],
        "Only information available up to the scoring hour is used, and the "
        "split is patient-grouped and time-ordered.",
        "hard",
    ),
    mk(
        "CL14", "Lending - limit increase", "classification",
        "customer accepts a credit-limit-increase offer",
        "scored before the offer is sent",
        "time-based split",
        "scaling within folds",
        [
            _f("current_limit", "float", "limit before the offer"),
            _f("avg_utilization_6m", "float", "mean utilisation over prior 6 months"),
            _f("months_since_last_increase", "int", "months since last increase"),
        ],
        False, "none", [],
        "All inputs are pre-offer; the outcome (acceptance) occurs after "
        "scoring. No leakage.",
        "easy",
    ),
    mk(
        "CL15", "Support - resolution time", "regression",
        "hours to resolve a support ticket",
        "scored at ticket creation",
        "random split by ticket",
        "encoders fit on train only",
        [
            _f("priority", "cat", "priority set at creation"),
            _f("product_area", "cat", "affected product area"),
            _f("customer_tier", "cat", "support tier of the customer"),
        ],
        False, "none", [],
        "Fields are all set at ticket creation; tickets are independent so a "
        "random split is appropriate.",
        "easy",
    ),
    mk(
        "CL16", "Energy - load forecasting", "regression",
        "electricity demand for the next hour",
        "forecast at the top of the current hour",
        "time-based split",
        "scaling fit on train only",
        [
            _f("load_lag_1h", "float", "demand one hour earlier"),
            _f("load_lag_24h", "float", "demand 24 hours earlier"),
            _f("temp_forecast_next_hour", "float",
               "weather-service temperature forecast for the next hour, issued "
               "before the prediction time"),
        ],
        False, "none", [],
        "A forecasted temperature issued before prediction time is legitimately "
        "available; it is a published forecast, not realised future data.",
        "hard",
    ),
]
