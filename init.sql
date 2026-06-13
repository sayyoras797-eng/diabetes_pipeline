-- PostgreSQL schema for the diabetes risk pipeline

CREATE TABLE IF NOT EXISTS patient_records (
    id              SERIAL PRIMARY KEY,
    age             INTEGER       NOT NULL,
    gender          VARCHAR(10)   NOT NULL,
    bmi             NUMERIC(5,2)  NOT NULL,
    glucose         NUMERIC(6,2)  NOT NULL,
    blood_pressure  NUMERIC(6,2)  NOT NULL,
    hba1c           NUMERIC(5,2)  NOT NULL,
    insulin         NUMERIC(6,2)  NOT NULL,
    skin_thickness  NUMERIC(5,2)  NOT NULL,
    pregnancies     INTEGER       NOT NULL,
    diabetes_risk   NUMERIC(6,4)  NOT NULL,
    label           SMALLINT      NOT NULL,
    recorded_at     TIMESTAMP     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patient_records_recorded_at ON patient_records (recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_patient_records_label ON patient_records (label);
