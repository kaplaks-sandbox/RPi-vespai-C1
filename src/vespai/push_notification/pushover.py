#!/usr/bin/env python3
"""
VespAI Push Notification Module - Pushover API Integration

This module handles push notifications for hornet detections using the Pushover service.
Provides rate limiting, cost tracking, and comprehensive error handling.

Author: Andre Jordaan
Version: 1.0
"""

import json
import os
import datetime
import logging
import requests
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

load_dotenv()

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")


def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


class PushoverMessage:
    """
    Pushover API client for sending push notifications.
    
    This class handles authentication, message sending, and error handling
    for the Pushover service used in VespAI hornet detection alerts.
    """
    
    def __init__(self, api_key: str, sender_name: str = "VespAI"):
        """
        Initialize the Pushover client.
        
        Args:
            api_key (str): Pushover API key or username:password format
            sender_name (str): Sender name that appears on push notifications
        """
        self.api_key = api_key
        self.sender_name = sender_name
        self.push_available = True

        # Parse API key format (username:password or token)
        if ":" in api_key:
            self.username, self.password = api_key.split(":", 1)
        else:
            # Use API key as password with empty username
            self.username = ""
            self.password = api_key

    def send_push(self, to: str, message: str, attachment: Optional[bytes] = None) -> Tuple[bool, float]:
        """
        Send a push notification via the Pushover API.
        
        Args:
            to (str): Recipient identifier (e.g., Pushover user key)
            message (str): Push notification content
            attachment (Optional[bytes]): Optional JPEG bytes to attach as image thumbnail
            
        Returns:
            Tuple[bool, float]: (success, cost) - Success status and message cost in EUR
        """
        if not self.push_available:
            logger.info(f"[Push notifications disabled] Would send: {message}")
            return False, 0.0

        url = "https://api.pushover.net/1/messages.json"

        try:
            logger.info(f"Sending push notification to {to}: {message[:50]}...")

            data = {
                "user": to,
                "token": self.api_key,
                "title": self.sender_name,
                "message": message,
            }

            logger.debug("Push notification payload: %s", json.dumps(data, indent=4))

            # Send request with 100 second timeout
            if attachment:
                files = {
                    'attachment': ('thumbnail.jpg', attachment, 'image/jpeg'),
                }
                response = requests.post(url, data=data, files=files, timeout=100)
            else:
                response = requests.post(url, data=data, timeout=100)
            
            if response.status_code != 200:  # OK
                error_msg = self._handle_error_response(response)
                logger.error("Push notification sending failed: %s", error_msg)
                return False, 0.0
            else:
                logger.info("✓ Push notification sent successfully (status: %d)", response.status_code)
                response_data = response.json()
                logger.debug("Push notification response: %s", json.dumps(response_data, indent=4))
                
                # Extract cost from response
                cost = self._extract_cost_from_response(response_data)
                return True, cost

        except requests.exceptions.RequestException as e:
            logger.error("Push notification request error: %s", e)
            return False, 0.0

    def _handle_error_response(self, response: requests.Response) -> str:
        """
        Handle and format error responses from the Pushover API.
        
        Args:
            response: The failed HTTP response
            
        Returns:
            str: Formatted error message
        """
        error_messages = {
            400: "Invalid input parameters",
            401: "Client ID or API key is invalid or inactive",
            402: "Insufficient funds in account",
            403: "Account not activated - contact support",
            404: "Resource not found",
            500: "Internal server error - contact Pushover support",
            502: "Bad gateway - contact Pushover support", 
            503: "Service unavailable - contact Pushover support",
            504: "Gateway timeout - contact Pushover support"
        }
        
        error_msg = error_messages.get(response.status_code, f"Unknown error (status: {response.status_code})")
        
        try:
            response_text = response.text
            logger.debug("Error response body: %s", response_text)
        except:
            response_text = "Unable to read response"
            
        return f"{error_msg}. Response: {response_text}"

    def _extract_cost_from_response(self, response_data: Dict[str, Any]) -> float:
        """
        Extract the push notification cost from the API response.
        
        Args:
            response_data: JSON response from Pushover API
            
        Returns:
            float: Message cost in EUR, or 0.0 if not found
        """
        # Try different possible cost field names
        cost_fields = ['price', 'cost', 'total_price', 'amount']
        
        for field in cost_fields:
            if field in response_data:
                try:
                    return float(response_data[field])
                except (ValueError, TypeError):
                    continue
                    
        return 0.0


class PushManager:
    """
    Push Notification Manager with rate limiting and statistics tracking.
    
    Manages push notification sending with configurable delays, cost tracking,
    and integration with VespAI statistics.
    """
    
    def __init__(self, 
                 api_key: str, 
                 phone_number: str,
                 sender_name: str = "VespAI",
                 delay_minutes: int = 5,
                 enabled: bool = True):
        """
        Initialize the Push Notification Manager.
        
        Args:
            api_key (str): Pushover API key
            phone_number (str): Target phone number for alerts
            sender_name (str): Push notification sender name
            delay_minutes (int): Minimum delay between push notifications
            enabled (bool): Whether push notification sending is enabled
        """
        self.phone_number = phone_number
        self.delay_minutes = delay_minutes
        self.enabled = enabled
        self.last_push_time: Optional[datetime.datetime] = None
        
        # Initialize push notification client
        if api_key and phone_number and enabled:
            self.client = PushoverMessage(api_key, sender_name)
        else:
            self.client = None
            logger.warning("Push notifications not configured - alerts disabled")
    
    def send_alert(self, message: str, force: bool = False, attachment: Optional[bytes] = None) -> Tuple[bool, str]:
        """
        Send a push notification alert with rate limiting.
        
        Args:
            message (str): Alert message to send
            force (bool): Whether to bypass rate limiting
            attachment (Optional[bytes]): Optional JPEG bytes to attach as image thumbnail
            
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        # Check if push notifications are enabled and configured
        if not self.enabled:
            return False, "Push notifications disabled in configuration"
            
        if not self.client:
            return False, "Push notification client not initialized"
            
        if not self.phone_number:
            return False, "No phone number configured"
        
        # Check rate limiting
        if not force and self.last_push_time is not None:
            time_since_last = (datetime.datetime.now() - self.last_push_time).total_seconds() / 60
            
            if time_since_last < self.delay_minutes:
                remaining = self.delay_minutes - time_since_last
                return False, f"Rate limited - next push notification allowed in {remaining:.1f} minutes"
        
        # Send the push notification
        success, cost = self.client.send_push(self.phone_number, message, attachment=attachment)
        
        if success:
            self.last_push_time = datetime.datetime.now()
            status_msg = f"Push notification sent successfully"
            if cost > 0:
                status_msg += f" (cost: {cost:.3f}€)"
            return True, status_msg
        else:
            return False, "Failed to send push notification"
    
    def create_hornet_alert(self, 
                          hornet_type: str, 
                          count: int, 
                          confidence: float,
                          frame_url: str) -> str:
        """
        Create a formatted hornet detection alert message.
        
        Args:
            hornet_type (str): Type of hornet ('velutina' or 'crabro')
            count (int): Number of hornets detected
            confidence (float): Detection confidence percentage
            frame_url (str): URL to view the detection frame
            
        Returns:
            str: Formatted alert message
        """
        time_str = datetime.datetime.now().strftime('%H:%M')
        
        if hornet_type == 'velutina':
            emoji = "⚠️"
            species = "Asian Hornet"
            urgency = "ALERT"
        else:
            emoji = "ℹ️"
            species = "European Hornet"
            urgency = "Info"
            
        plural = "s" if count > 1 else ""
        
        message = f"{emoji} {urgency}: {count} {species}{plural} detected at {time_str}"
        
        if confidence > 0:
            message += f" ({confidence:.1f}% confidence)"
            
        if frame_url:
            message += f". View: {frame_url}"
            
        return message


def create_push_manager_from_env() -> Optional[PushManager]:
    """
    Create a push notification manager from environment variables.
    
    Expected environment variables:
    - PUSHOVER_TOKEN: Pushover API token
    - PUSHOVER_USER: Pushover user key
    - PUSHOVER_SENDER: Push sender name (optional)
    - PUSH_DELAY_MINUTES: Delay between notifications (optional)
    - ENABLE_PUSH: Whether push notifications are enabled (optional)
    
    Returns:
        PushManager: Configured push manager or None if not properly configured
    """
    api_key = os.getenv("PUSHOVER_TOKEN", "")
    phone_number = os.getenv("PUSHOVER_USER", "")
    sender_name = os.getenv("PUSHOVER_SENDER", os.getenv("VESPAI_NAME", "VespAI"))
    delay_minutes = int(os.getenv("PUSH_DELAY_MINUTES", "5"))
    enabled = os.getenv("ENABLE_PUSH", "true").lower() == "true"
    
    if not api_key:
        logger.warning("PUSHOVER_TOKEN not set - push alerts disabled")
        return None
        
    if not phone_number:
        logger.warning("PUSHOVER_USER not set - push alerts disabled") 
        return None
    
    try:
        return PushManager(
            api_key=api_key,
            phone_number=phone_number,
            sender_name=sender_name,
            delay_minutes=delay_minutes,
            enabled=enabled
        )
    except Exception as e:
        logger.error("Failed to initialize push manager: %s", e)
        return None