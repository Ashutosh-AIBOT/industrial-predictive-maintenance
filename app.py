import os

from dashboard_core import ProjectConfig, run_project_app


CONFIG = ProjectConfig(
    key="Project 02",
    title="Predictive Maintenance",
    subtitle="Industrial asset failure prediction for proactive service planning",
    icon="🛠️",
    domain="Manufacturing and industrial operations",
    objective="Forecast machine failures before breakdown events using sensor-driven features.",
    business_value="Reduces unplanned downtime, lowers maintenance costs, and improves SLA compliance.",
    prediction_label="Failure Risk",
    highlights=[
        "Combines reliability engineering context with ML-based failure forecasting.",
        "Enables preventive action windows through fast single and batch inference.",
        "Compares multiple algorithms for robust plant-floor deployment decisions.",
        "Integrates charts and notebook lineage for audit-ready model storytelling.",
        "Reflects production thinking with monitoring-ready dashboard structure.",
    ],
)


run_project_app(CONFIG, os.path.dirname(__file__))
