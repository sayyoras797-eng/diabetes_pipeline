import os
import sys
import smtplib
from email.mime.text import MIMEText
import psycopg2
from dagster import op, job, schedule, Definitions, RunRequest

sys.path.append('/app')
from diabetes_pipeline.generator.data_generator import generate_batch


@op
def generate_patients(context):
    records = generate_batch(n=50)
    context.log.info(f"Generated {len(records)} patient records")
    return records


@op
def write_to_postgres(context, records):
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "diabetes_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )
    cur = conn.cursor()
    sql = """
        INSERT INTO patient_records
        (age, gender, bmi, glucose, blood_pressure, hba1c,
         insulin, skin_thickness, pregnancies,
         physical_activity, smoking_status, alcohol_consumption, family_history,
         diabetes_risk, label, recorded_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    for r in records:
        cur.execute(sql, (
            r["age"], r["gender"], r["bmi"], r["glucose"],
            r["blood_pressure"], r["hba1c"], r["insulin"],
            r["skin_thickness"], r["pregnancies"],
            r["physical_activity"], r["smoking_status"],
            r["alcohol_consumption"], r["family_history"],
            r["diabetes_risk"], r["label"], r["recorded_at"]
        ))
    conn.commit()
    cur.close()
    conn.close()
    context.log.info(f"Wrote {len(records)} records to PostgreSQL")
    return len(records)


@op
def log_summary(context, count):
    context.log.info(f"Pipeline complete. Total inserted: {count}")


HIGH_RISK_THRESHOLD = 0.8


@op
def send_high_risk_alert(context, records):
    high_risk = [r for r in records if r["diabetes_risk"] >= HIGH_RISK_THRESHOLD]
    if not high_risk:
        context.log.info("No high-risk patients in this batch — no alert sent")
        return 0

    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        context.log.info(
            f"{len(high_risk)} high-risk patient(s) found, but SMTP_HOST is not "
            "configured — skipping email alert"
        )
        return len(high_risk)

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_from = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    alert_to = os.getenv("ALERT_EMAIL_TO")

    if not alert_to:
        context.log.warning("ALERT_EMAIL_TO is not set — skipping email alert")
        return len(high_risk)

    lines = [
        f"Age {r['age']}, {r['gender']}, glucose={r['glucose']}, "
        f"BMI={r['bmi']}, HbA1c={r['hba1c']}, risk={r['diabetes_risk']:.2f}"
        for r in high_risk
    ]
    body = (
        f"{len(high_risk)} high-risk patient(s) (risk >= {HIGH_RISK_THRESHOLD}) "
        f"detected in the latest batch:\n\n" + "\n".join(lines)
    )
    msg = MIMEText(body)
    msg["Subject"] = f"[Diabetes Pipeline] {len(high_risk)} high-risk patient(s) detected"
    msg["From"] = alert_from
    msg["To"] = alert_to

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.sendmail(alert_from, [alert_to], msg.as_string())

    context.log.info(f"Sent high-risk alert email for {len(high_risk)} patient(s)")
    return len(high_risk)


@job(name="diabetes_hourly_pipeline")
def diabetes_job():
    records = generate_patients()
    count = write_to_postgres(records)
    send_high_risk_alert(records)
    log_summary(count)


@schedule(
    cron_schedule="0 9-17 * * 1-5",
    job=diabetes_job,
    execution_timezone="Asia/Tashkent",
)
def diabetes_hourly_schedule(context):
    return RunRequest(run_key=None, run_config={})


defs = Definitions(
    jobs=[diabetes_job],
    schedules=[diabetes_hourly_schedule],
)