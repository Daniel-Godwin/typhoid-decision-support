"""WTForms definitions. Server-side validation for every submitted form."""
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    EmailField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    NumberRange,
    Optional,
    ValidationError,
)

from .models import Role
from .security import password_problems


class StrongPassword:
    """Applies the shared password policy inside WTForms validation."""

    def __init__(self, minimum: int = 10):
        self.minimum = minimum

    def __call__(self, form, field):
        issues = password_problems(field.data or "", self.minimum)
        if issues:
            raise ValidationError("Password " + "; ".join(issues) + ".")


class LoginForm(FlaskForm):
    email = EmailField(
        "Email address",
        validators=[
            DataRequired("Enter your email address."),
            # No DNS deliverability lookup: a sign-in form must not depend on
            # network resolution, and internal domains are legitimate.
            Email("Enter a valid email address.", check_deliverability=False),
        ],
        render_kw={"autocomplete": "username", "placeholder": "you@clinic.org"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired("Enter your password.")],
        render_kw={"autocomplete": "current-password", "placeholder": "••••••••••"},
    )
    remember = BooleanField("Keep me signed in")
    submit = SubmitField("Sign in")


class ChangePasswordForm(FlaskForm):
    current_password = PasswordField(
        "Current password",
        validators=[DataRequired("Enter your current password.")],
        render_kw={"autocomplete": "current-password"},
    )
    new_password = PasswordField(
        "New password",
        validators=[DataRequired("Choose a new password."), StrongPassword()],
        render_kw={"autocomplete": "new-password"},
    )
    confirm_password = PasswordField(
        "Confirm new password",
        validators=[
            DataRequired("Confirm your new password."),
            EqualTo("new_password", "Passwords do not match."),
        ],
        render_kw={"autocomplete": "new-password"},
    )
    submit = SubmitField("Update password")


class ProfileForm(FlaskForm):
    full_name = StringField(
        "Full name", validators=[DataRequired("Enter your name."), Length(max=160)]
    )
    facility = StringField("Facility", validators=[Optional(), Length(max=160)])
    submit = SubmitField("Save changes")


class UserForm(FlaskForm):
    """Administrator creating or editing an account."""

    full_name = StringField(
        "Full name", validators=[DataRequired("Enter the user's name."), Length(max=160)]
    )
    email = EmailField(
        "Email address",
        validators=[
            DataRequired("Enter an email address."),
            Email("Enter a valid email address.", check_deliverability=False),
        ],
    )
    role = SelectField(
        "Role",
        choices=[(r, Role.LABELS[r]) for r in Role.ALL],
        validators=[DataRequired("Select a role.")],
    )
    facility = StringField("Facility", validators=[Optional(), Length(max=160)])
    is_active_account = BooleanField("Account enabled", default=True)
    password = PasswordField(
        "Temporary password",
        validators=[Optional(), StrongPassword()],
        description="Leave blank when editing to keep the existing password.",
    )
    must_change_password = BooleanField("Require password change at next sign-in", default=True)
    submit = SubmitField("Save user")

    def __init__(self, *args, editing_user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.editing_user = editing_user
        if editing_user is None:
            self.password.validators = [DataRequired("Set a temporary password."), StrongPassword()]

    def validate_email(self, field):
        from .models import User

        existing = User.query.filter(User.email == field.data.lower().strip()).first()
        if existing and (self.editing_user is None or existing.id != self.editing_user.id):
            raise ValidationError("An account with this email already exists.")


class PatientForm(FlaskForm):
    patient_code = StringField(
        "Patient code",
        validators=[DataRequired("Enter a patient code."), Length(max=40)],
        description="Your facility's identifier for this patient.",
    )
    full_name = StringField(
        "Full name", validators=[DataRequired("Enter the patient's name."), Length(max=160)]
    )
    age = IntegerField(
        "Age (years)",
        validators=[Optional(), NumberRange(min=0, max=120, message="Age must be 0–120.")],
    )
    gender = SelectField(
        "Gender",
        choices=[("", "Select…"), ("Male", "Male"), ("Female", "Female")],
        validators=[Optional()],
    )
    location = SelectField(
        "Setting",
        choices=[("", "Select…"), ("Urban", "Urban"), ("Rural", "Rural"), ("Endemic", "Endemic")],
        validators=[Optional()],
    )
    contact = StringField("Contact", validators=[Optional(), Length(max=80)])
    notes = TextAreaField("Clinical notes", validators=[Optional(), Length(max=2000)])
    submit = SubmitField("Save patient")

    def __init__(self, *args, editing_patient=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.editing_patient = editing_patient

    def validate_patient_code(self, field):
        from .models import Patient

        existing = Patient.query.filter(Patient.patient_code == field.data.strip()).first()
        if existing and (self.editing_patient is None or existing.id != self.editing_patient.id):
            raise ValidationError("This patient code is already in use.")


class AssessmentMetaForm(FlaskForm):
    """Non-clinical options attached to an assessment submission."""

    patient_id = SelectField("Patient record", choices=[], validators=[Optional()])
    target_mode = SelectField(
        "Model",
        choices=[
            ("binary", "Diagnosis — Typhoid / No Typhoid"),
            ("multiclass", "Severity stratification"),
        ],
        default="binary",
    )
    submit = SubmitField("Run assessment")
