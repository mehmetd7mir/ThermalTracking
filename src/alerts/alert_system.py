"""
Alert System Module
--------------------
Send notifications when threats are detected.

Supported channels:
    - Telegram
    - Webhook (Slack, Discord, custom)
    - Email (SMTP)
    - Audio alarm (local beep)

How to use:
    alert = AlertSystem()
    alert.add_telegram(token="...", chat_id="...")
    alert.send("Drone detected in restricted area!")

Author: Mehmet Demir
"""

import json
import time
import threading
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import queue


@dataclass
class AlertMessage:
    """Container for alert message"""
    level: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data
        }


class AlertChannel(ABC):
    """Base class for alert channels"""
    
    @abstractmethod
    def send(self, alert: AlertMessage) -> bool:
        """Send alert through this channel. Return True if success."""
        pass
    
    @abstractmethod
    def test(self) -> bool:
        """Test if channel is working."""
        pass


class TelegramChannel(AlertChannel):
    """
    Send alerts to Telegram chat.
    
    To setup:
        1. Create bot with @BotFather
        2. Get token from BotFather
        3. Start chat with your bot
        4. Get chat_id from https://api.telegram.org/bot<TOKEN>/getUpdates
    """
    
    def __init__(self, token: str, chat_id: str):
        """
        Initialize telegram channel.
        
        Args:
            token: bot token from BotFather
            chat_id: chat id to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send(self, alert: AlertMessage) -> bool:
        """Send message to telegram."""
        try:
            import requests
        except ImportError:
            print("requests package needed for telegram. Install: pip install requests")
            return False
        
        # format message
        level_emoji = {
            "info": "i",
            "warning": "!",
            "critical": "!!!"
        }
        emoji = level_emoji.get(alert.level, "")
        
        text = f"[{emoji}] {alert.title}\n\n{alert.message}"
        
        if alert.data:
            text += "\n\nDetails:\n"
            for key, value in alert.data.items():
                text += f"  {key}: {value}\n"
        
        # send request
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def test(self) -> bool:
        """Send test message."""
        test_alert = AlertMessage(
            level="info",
            title="Test Alert",
            message="This is test message from ThermalTracking"
        )
        return self.send(test_alert)
    
    def send_image(self, image_path: str, caption: str = "") -> bool:
        """Send image to telegram."""
        try:
            import requests
        except ImportError:
            return False
        
        url = f"{self.base_url}/sendPhoto"
        
        try:
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": self.chat_id, "caption": caption}
                response = requests.post(url, files=files, data=data, timeout=30)
                return response.status_code == 200
        except Exception as e:
            print(f"Telegram image send error: {e}")
            return False


class WebhookChannel(AlertChannel):
    """
    Send alerts to webhook URL.
    
    Works with Slack, Discord, custom endpoints.
    Sends JSON POST request.
    """
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        format_func: Optional[Callable] = None
    ):
        """
        Initialize webhook.
        
        Args:
            url: webhook URL
            headers: optional HTTP headers
            format_func: optional function to format message
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.format_func = format_func
    
    def send(self, alert: AlertMessage) -> bool:
        """Send alert to webhook."""
        try:
            import requests
        except ImportError:
            print("requests package needed")
            return False
        
        # format payload
        if self.format_func:
            payload = self.format_func(alert)
        else:
            payload = alert.to_dict()
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            return response.status_code in [200, 201, 204]
        except Exception as e:
            print(f"Webhook error: {e}")
            return False
    
    def test(self) -> bool:
        """Test webhook connection."""
        test_alert = AlertMessage(
            level="info",
            title="Webhook Test",
            message="Testing connection from ThermalTracking"
        )
        return self.send(test_alert)


class SlackChannel(WebhookChannel):
    """
    Slack specific webhook channel.
    
    Uses Slack message format with blocks.
    """
    
    def __init__(self, webhook_url: str):
        super().__init__(webhook_url)
        self.format_func = self._format_slack
    
    def _format_slack(self, alert: AlertMessage) -> Dict:
        """Format for Slack incoming webhook."""
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9800",
            "critical": "#f44336"
        }
        
        return {
            "attachments": [{
                "color": color_map.get(alert.level, "#808080"),
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in alert.data.items()
                ],
                "ts": int(alert.timestamp)
            }]
        }


class DiscordChannel(WebhookChannel):
    """Discord webhook channel."""
    
    def __init__(self, webhook_url: str):
        super().__init__(webhook_url)
        self.format_func = self._format_discord
    
    def _format_discord(self, alert: AlertMessage) -> Dict:
        """Format for Discord webhook."""
        color_map = {
            "info": 0x36a64f,
            "warning": 0xff9800,
            "critical": 0xf44336
        }
        
        return {
            "embeds": [{
                "title": alert.title,
                "description": alert.message,
                "color": color_map.get(alert.level, 0x808080),
                "fields": [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in alert.data.items()
                ],
                "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat()
            }]
        }


class EmailChannel(AlertChannel):
    """
    Send alerts via email.
    
    Uses SMTP to send emails.
    """
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
    
    def send(self, alert: AlertMessage) -> bool:
        """Send email alert."""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = f"[{alert.level.upper()}] {alert.title}"
            
            body = f"{alert.message}\n\n"
            if alert.data:
                body += "Details:\n"
                for k, v in alert.data.items():
                    body += f"  {k}: {v}\n"
            
            body += f"\nTimestamp: {datetime.fromtimestamp(alert.timestamp)}"
            
            msg.attach(MIMEText(body, "plain"))
            
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            
            server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email error: {e}")
            return False
    
    def test(self) -> bool:
        test_alert = AlertMessage(
            level="info",
            title="Email Test",
            message="Testing email from ThermalTracking"
        )
        return self.send(test_alert)


class AudioChannel(AlertChannel):
    """
    Play audio alarm locally.
    
    Simple beep or sound file playback.
    """
    
    def __init__(self, sound_file: Optional[str] = None):
        """
        Args:
            sound_file: path to audio file or None for system beep
        """
        self.sound_file = sound_file
    
    def send(self, alert: AlertMessage) -> bool:
        """Play alert sound."""
        try:
            if self.sound_file:
                # try to play file using playsound or pygame
                try:
                    from playsound import playsound
                    playsound(self.sound_file)
                    return True
                except ImportError:
                    pass
            
            # fallback to system beep
            import sys
            if sys.platform == "win32":
                import winsound
                frequency = 1000 if alert.level == "critical" else 500
                duration = 500 if alert.level == "critical" else 200
                winsound.Beep(frequency, duration)
            else:
                # unix beep
                print("\a", end="", flush=True)
            
            return True
            
        except Exception as e:
            print(f"Audio error: {e}")
            return False
    
    def test(self) -> bool:
        return self.send(AlertMessage(level="info", title="", message=""))


class AlertSystem:
    """
    Main alert system that manage multiple channels.
    
    Example:
        alerts = AlertSystem()
        alerts.add_telegram("token", "chat_id")
        alerts.add_webhook("https://hooks.slack.com/...")
        
        # send alert
        alerts.send(
            level="critical",
            title="Drone Detected",
            message="Unauthorized drone in Zone A",
            data={"class": "drone", "confidence": 0.95}
        )
    """
    
    def __init__(self, async_mode: bool = True):
        """
        Initialize alert system.
        
        Args:
            async_mode: if True, send alerts in background thread
        """
        self.channels: Dict[str, AlertChannel] = {}
        self.async_mode = async_mode
        self.alert_history: List[AlertMessage] = []
        self.max_history = 1000
        
        # rate limiting
        self.min_interval = 1.0  # seconds between same alerts
        self.last_alert_time = 0.0
        
        # async queue
        if async_mode:
            self.alert_queue: queue.Queue = queue.Queue()
            self.worker_thread = threading.Thread(
                target=self._worker,
                daemon=True
            )
            self.worker_thread.start()
    
    def add_channel(self, name: str, channel: AlertChannel):
        """Add alert channel."""
        self.channels[name] = channel
    
    def add_telegram(self, token: str, chat_id: str, name: str = "telegram"):
        """Add telegram channel."""
        self.channels[name] = TelegramChannel(token, chat_id)
    
    def add_webhook(self, url: str, name: str = "webhook"):
        """Add generic webhook."""
        self.channels[name] = WebhookChannel(url)
    
    def add_slack(self, webhook_url: str, name: str = "slack"):
        """Add Slack webhook."""
        self.channels[name] = SlackChannel(webhook_url)
    
    def add_discord(self, webhook_url: str, name: str = "discord"):
        """Add Discord webhook."""
        self.channels[name] = DiscordChannel(webhook_url)
    
    def remove_channel(self, name: str):
        """Remove channel."""
        if name in self.channels:
            del self.channels[name]
    
    def send(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict] = None,
        channels: Optional[List[str]] = None
    ) -> bool:
        """
        Send alert to all or specific channels.
        
        Args:
            level: 'info', 'warning', or 'critical'
            title: alert title
            message: alert body
            data: additional data dict
            channels: specific channels to use (all if None)
        
        Returns:
            True if at least one channel succeeded
        """
        # rate limiting
        now = time.time()
        if now - self.last_alert_time < self.min_interval:
            return False
        self.last_alert_time = now
        
        alert = AlertMessage(
            level=level,
            title=title,
            message=message,
            data=data or {}
        )
        
        # add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # determine which channels to use
        target_channels = channels or list(self.channels.keys())
        
        if self.async_mode:
            # queue for background processing
            self.alert_queue.put((alert, target_channels))
            return True
        else:
            # send synchronously
            return self._send_sync(alert, target_channels)
    
    def _send_sync(self, alert: AlertMessage, channel_names: List[str]) -> bool:
        """Send alert synchronously."""
        success = False
        for name in channel_names:
            if name in self.channels:
                try:
                    if self.channels[name].send(alert):
                        success = True
                except Exception as e:
                    print(f"Channel {name} failed: {e}")
        return success
    
    def _worker(self):
        """Background worker for async alerts."""
        while True:
            try:
                alert, channels = self.alert_queue.get()
                self._send_sync(alert, channels)
                self.alert_queue.task_done()
            except Exception as e:
                print(f"Alert worker error: {e}")
    
    def test_all(self) -> Dict[str, bool]:
        """Test all channels."""
        results = {}
        for name, channel in self.channels.items():
            try:
                results[name] = channel.test()
            except Exception as e:
                print(f"Test failed for {name}: {e}")
                results[name] = False
        return results
    
    def get_history(self, count: int = 10) -> List[Dict]:
        """Get recent alert history."""
        recent = self.alert_history[-count:]
        return [a.to_dict() for a in recent]


# helper function for quick alerts
def send_telegram_alert(
    token: str,
    chat_id: str,
    message: str,
    title: str = "Alert"
) -> bool:
    """Quick function to send telegram alert."""
    channel = TelegramChannel(token, chat_id)
    alert = AlertMessage(level="warning", title=title, message=message)
    return channel.send(alert)


# test
if __name__ == "__main__":
    # create alert system
    alerts = AlertSystem(async_mode=False)
    
    # add test webhook (httpbin for testing)
    alerts.add_webhook("https://httpbin.org/post", name="test")
    
    # send test alert
    success = alerts.send(
        level="critical",
        title="Test Alert",
        message="This is test from ThermalTracking",
        data={
            "class": "drone",
            "confidence": 0.95,
            "zone": "restricted_area"
        }
    )
    
    print(f"Alert sent: {success}")
    print(f"History: {alerts.get_history()}")
