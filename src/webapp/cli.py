"""Command-line administration, used locally and during deployment."""
from __future__ import annotations

import secrets
import string

import click
from flask.cli import with_appcontext

from .extensions import db
from .models import Assessment, AuditLog, Patient, Role, User


def _generate_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@click.command("init-db")
@with_appcontext
def init_db():
    """Create any missing tables. Safe to run repeatedly."""
    db.create_all()
    click.echo("Database tables are present.")


@click.command("create-admin")
@click.option("--email", prompt=True)
@click.option("--name", prompt="Full name")
@click.option("--password", default=None, help="Omit to generate one.")
@with_appcontext
def create_admin(email, name, password):
    """Create an administrator account."""
    email = email.lower().strip()
    if User.query.filter_by(email=email).first():
        raise click.ClickException(f"An account already exists for {email}")
    generated = password is None
    password = password or _generate_password()
    user = User(
        full_name=name.strip(), email=email, role=Role.ADMIN,
        is_active_account=True, must_change_password=generated,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    click.echo(f"Administrator created: {email}")
    if generated:
        click.echo(f"Temporary password: {password}")
        click.echo("This must be changed at first sign-in.")


@click.command("bootstrap")
@with_appcontext
def bootstrap():
    """Prepare a fresh deployment: tables plus the initial administrator.

    Reads BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD. Idempotent, so it
    is safe as a Render release command.
    """
    from flask import current_app

    db.create_all()
    click.echo("Database tables are present.")

    email = current_app.config.get("BOOTSTRAP_ADMIN_EMAIL")
    password = current_app.config.get("BOOTSTRAP_ADMIN_PASSWORD")
    if not email:
        click.echo("BOOTSTRAP_ADMIN_EMAIL not set; skipping administrator creation.")
        return
    email = email.lower().strip()
    if User.query.filter_by(email=email).first():
        click.echo(f"Administrator {email} already exists; nothing to do.")
        return
    generated = not password
    password = password or _generate_password()
    user = User(
        full_name="System Administrator", email=email, role=Role.ADMIN,
        is_active_account=True, must_change_password=True,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    click.echo(f"Administrator created: {email}")
    if generated:
        click.echo(f"Temporary password: {password}")


@click.command("seed-demo")
@click.option("--force", is_flag=True, help="Seed even if accounts already exist.")
@with_appcontext
def seed_demo(force):
    """Create demonstration accounts and patients for local testing."""
    if User.query.count() and not force:
        click.echo("Users already exist. Re-run with --force to seed anyway.")
        return

    accounts = [
        ("admin@example.com", "System Administrator", Role.ADMIN, "AdminPass2026"),
        ("clinician@example.com", "Dr Amina Bello", Role.CLINICIAN, "ClinicPass2026"),
        ("auditor@example.com", "Grace Okonkwo", Role.AUDITOR, "AuditPass2026"),
    ]
    created = {}
    for email, name, role, password in accounts:
        user = User.query.filter_by(email=email).first()
        if user is None:
            user = User(
                full_name=name, email=email, role=role,
                facility="Central Health Centre",
                is_active_account=True, must_change_password=False,
            )
            user.set_password(password)
            db.session.add(user)
        created[role] = user
    db.session.commit()

    clinician = created.get(Role.CLINICIAN) or User.query.filter_by(role=Role.CLINICIAN).first()
    samples = [
        ("PT-1001", "Ibrahim Musa", 34, "Male", "Urban"),
        ("PT-1002", "Ngozi Eze", 27, "Female", "Rural"),
        ("PT-1003", "Samuel Adeyemi", 9, "Male", "Endemic"),
        ("PT-1004", "Fatima Sani", 52, "Female", "Urban"),
    ]
    for code, name, age, gender, location in samples:
        if Patient.query.filter_by(patient_code=code).first():
            continue
        db.session.add(Patient(
            patient_code=code, full_name=name, age=age, gender=gender,
            location=location, created_by_id=clinician.id,
        ))
    db.session.commit()

    click.echo("Demonstration data seeded:")
    for email, _, role, password in accounts:
        click.echo(f"  {Role.LABELS[role]:14s} {email:28s} {password}")
    click.echo(f"  {Patient.query.count()} patient records")


@click.command("stats")
@with_appcontext
def stats():
    """Print a summary of stored data."""
    click.echo(f"users:       {User.query.count()}")
    click.echo(f"patients:    {Patient.query.count()}")
    click.echo(f"assessments: {Assessment.query.count()}")
    click.echo(f"audit rows:  {AuditLog.query.count()}")


def register(app):
    app.cli.add_command(init_db)
    app.cli.add_command(create_admin)
    app.cli.add_command(bootstrap)
    app.cli.add_command(seed_demo)
    app.cli.add_command(stats)
