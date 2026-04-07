"""
User Schemas
==============
Request/response schemas for user-related endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50, examples=["john_doe"])
    email: EmailStr = Field(..., examples=["john@example.com"])
    password: str = Field(..., min_length=6, max_length=100, examples=["securePass123"])
    full_name: Optional[str] = Field(None, max_length=100, examples=["John Doe"])
    role: Optional[str] = Field("operator", pattern="^(admin|operator)$")


class UserResponse(BaseModel):
    """Schema for user data in responses (never includes password)."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for login response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
