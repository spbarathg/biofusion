"""
ADVANCED ALERT SYSTEM
====================

Multi-channel alert system for Smart Ape Mode:
- Telegram bot notifications
- Discord webhook alerts  
- Email notifications
- Real-time status updates
- Alert priority management
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os
import asyncio

# Conditional email imports
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    # Create mock classes
    class MimeText:
        def __init__(self, *args, **kwargs): pass
    class MimeMultipart:
        def __init__(self, *args, **kwargs): pass
        def attach(self, *args): pass
        def as_string(self): return ""
    class smtplib:
        class SMTP:
            def __init__(self, *args, **kwargs): pass
            def starttls(self): pass
            def login(self, *args): pass
            def send_message(self, *args): pass
            def quit(self): pass

class AlertPriority(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    ALL = "all"

@dataclass
class Alert:
    """Alert message structure"""
    alert_id: str
    title: str
    message: str
    priority: AlertPriority
    timestamp: datetime
    channels: List[AlertChannel]
    data: Dict[str, Any]

class AdvancedAlertSystem:
    """Advanced multi-channel alert system"""
    
    def __init__(self):
        self.logger = logging.getLogger("AdvancedAlerts")
        
        # Configuration
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.alert_email = os.getenv("ALERT_EMAIL")
        
        # Alert queue and history
        self.alert_queue = asyncio.Queue()
        self.alert_history = []
        self.max_history = 1000
        
        # Rate limiting
        self.rate_limits = {
            AlertChannel.TELEGRAM: {'count': 0, 'window_start': datetime.now()},
            AlertChannel.DISCORD: {'count': 0, 'window_start': datetime.now()},
            AlertChannel.EMAIL: {'count': 0, 'window_start': datetime.now()}
        }
        self.max_alerts_per_minute = 10
        
        # Alert templates
        self.templates = {
            'profit_milestone': {
                'title': '🎯 Profit Milestone Reached',
                'template': 'Smart Ape Mode reached ${profit:.2f} profit! ROI: {roi:.1f}%'
            },
            'kill_switch_triggered': {
                'title': '🚨 Kill Switch Activated',
                'template': 'Emergency kill switch triggered: {reason}'
            },
            'rug_detected': {
                'title': '🔍 Rug Pull Detected',
                'template': 'Rug pull detected for token {token_symbol} ({token_address})'
            },
            'sniper_detected': {
                'title': '🥷 Sniper Activity',
                'template': 'Sniper/MEV bot detected: {threat_type} - Confidence: {confidence:.1%}'
            },
            'system_health': {
                'title': '⚡ System Health Alert',
                'template': 'System health issue: {issue} - Status: {status}'
            }
        }
        
        # Processing active flag
        self.processing_active = False
        
    async def initialize(self):
        """Initialize alert system"""
        
        self.logger.info("📡 Initializing Advanced Alert System")
        
        # Validate configurations
        await self._validate_configurations()
        
        # Start alert processing
        await self._start_alert_processing()
        
        self.logger.info("✅ Advanced Alert System online")
        
    async def _validate_configurations(self):
        """Validate alert system configurations"""
        
        warnings = []
        
        # Telegram validation
        if not self.telegram_bot_token or not self.telegram_chat_id:
            warnings.append("Telegram configuration missing")
        
        # Discord validation
        if not self.discord_webhook_url:
            warnings.append("Discord webhook URL missing")
        
        # Email validation
        if not self.email_user or not self.email_password or not self.alert_email:
            warnings.append("Email configuration missing")
        
        if warnings:
            self.logger.warning(f"⚠️ Alert configuration warnings: {', '.join(warnings)}")
        
    async def _start_alert_processing(self):
        """Start background alert processing"""
        
        self.processing_active = True
        asyncio.create_task(self._process_alert_queue())
        
    async def _process_alert_queue(self):
        """Process alerts from queue"""
        
        while self.processing_active:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Process alert
                await self._send_alert(alert)
                
                # Add to history
                self.alert_history.append(alert)
                
                # Maintain history size
                if len(self.alert_history) > self.max_history:
                    self.alert_history = self.alert_history[-self.max_history:]
                
                # Mark task done
                self.alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"💥 Alert processing error: {e}")
    
    async def send_alert(self, title: str, message: str, 
                        priority: AlertPriority = AlertPriority.INFO,
                        channels: List[AlertChannel] = None,
                        data: Dict[str, Any] = None):
        """Send alert via specified channels"""
        
        if channels is None:
            channels = [AlertChannel.ALL]
        
        if data is None:
            data = {}
        
        # Create alert object
        alert = Alert(
            alert_id=f"alert_{int(datetime.now().timestamp())}",
            title=title,
            message=message,
            priority=priority,
            timestamp=datetime.now(),
            channels=channels,
            data=data
        )
        
        # Add to queue
        await self.alert_queue.put(alert)
        
    async def send_templated_alert(self, template_name: str, 
                                  priority: AlertPriority = AlertPriority.INFO,
                                  channels: List[AlertChannel] = None,
                                  **kwargs):
        """Send alert using predefined template"""
        
        if template_name not in self.templates:
            self.logger.error(f"💥 Unknown template: {template_name}")
            return
        
        template = self.templates[template_name]
        
        try:
            # Format message with provided data
            formatted_message = template['template'].format(**kwargs)
            
            await self.send_alert(
                title=template['title'],
                message=formatted_message,
                priority=priority,
                channels=channels,
                data=kwargs
            )
            
        except KeyError as e:
            self.logger.error(f"💥 Template formatting error: Missing key {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert to all specified channels"""
        
        # Check rate limits
        if not await self._check_rate_limits(alert):
            self.logger.warning(f"⚠️ Rate limit exceeded for alert: {alert.title}")
            return
        
        # Determine target channels
        target_channels = []
        if AlertChannel.ALL in alert.channels:
            target_channels = [AlertChannel.TELEGRAM, AlertChannel.DISCORD, AlertChannel.EMAIL]
        else:
            target_channels = alert.channels
        
        # Send to each channel
        tasks = []
        
        if AlertChannel.TELEGRAM in target_channels:
            tasks.append(self._send_telegram_alert(alert))
        
        if AlertChannel.DISCORD in target_channels:
            tasks.append(self._send_discord_alert(alert))
        
        if AlertChannel.EMAIL in target_channels:
            tasks.append(self._send_email_alert(alert))
        
        # Execute all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_rate_limits(self, alert: Alert) -> bool:
        """Check if alert exceeds rate limits"""
        
        current_time = datetime.now()
        
        # Reset rate limit windows if needed
        for channel in self.rate_limits:
            rate_limit = self.rate_limits[channel]
            
            # Reset if window expired (1 minute)
            if (current_time - rate_limit['window_start']).total_seconds() > 60:
                rate_limit['count'] = 0
                rate_limit['window_start'] = current_time
        
        # Check limits for each target channel
        target_channels = alert.channels
        if AlertChannel.ALL in target_channels:
            target_channels = [AlertChannel.TELEGRAM, AlertChannel.DISCORD, AlertChannel.EMAIL]
        
        for channel in target_channels:
            if channel in self.rate_limits:
                if self.rate_limits[channel]['count'] >= self.max_alerts_per_minute:
                    # Allow emergency alerts to bypass rate limits
                    if alert.priority != AlertPriority.EMERGENCY:
                        return False
        
        # Increment counters
        for channel in target_channels:
            if channel in self.rate_limits:
                self.rate_limits[channel]['count'] += 1
        
        return True
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send alert via Telegram"""
        
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        try:
            # Format message
            message = self._format_telegram_message(alert)
            
            # Telegram API URL
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            # Prepare payload
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"📱 Telegram alert sent: {alert.title}")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"💥 Telegram alert failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"💥 Telegram alert error: {e}")
    
    async def _send_discord_alert(self, alert: Alert):
        """Send alert via Discord webhook"""
        
        if not self.discord_webhook_url:
            return
        
        try:
            # Format Discord embed
            embed = self._format_discord_embed(alert)
            
            # Prepare payload
            payload = {
                'embeds': [embed]
            }
            
            # Send request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook_url, json=payload) as response:
                    if response.status in [200, 204]:
                        self.logger.info(f"💬 Discord alert sent: {alert.title}")
                    else:
                        error_text = await response.text()
                        self.logger.error(f"💥 Discord alert failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"💥 Discord alert error: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        
        if not all([self.email_user, self.email_password, self.alert_email]):
            return
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.alert_email
            msg['Subject'] = f"[Smart Ape Mode] {alert.title}"
            
            # Format email body
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, self.alert_email, text)
            server.quit()
            
            self.logger.info(f"📧 Email alert sent: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"💥 Email alert error: {e}")
    
    def _format_telegram_message(self, alert: Alert) -> str:
        """Format message for Telegram"""
        
        # Priority emoji
        priority_emoji = {
            AlertPriority.INFO: "ℹ️",
            AlertPriority.WARNING: "⚠️", 
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.EMERGENCY: "🔥"
        }
        
        emoji = priority_emoji.get(alert.priority, "ℹ️")
        
        message = f"{emoji} *{alert.title}*\n\n"
        message += f"{alert.message}\n\n"
        message += f"⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"🔗 Priority: {alert.priority.value.upper()}"
        
        # Add data if present
        if alert.data:
            message += f"\n\n📊 *Details:*\n"
            for key, value in alert.data.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_percent') or key.endswith('_rate'):
                        message += f"• {key.replace('_', ' ').title()}: {value:.1f}%\n"
                    else:
                        message += f"• {key.replace('_', ' ').title()}: {value}\n"
                else:
                    message += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        return message
    
    def _format_discord_embed(self, alert: Alert) -> Dict[str, Any]:
        """Format embed for Discord"""
        
        # Priority colors
        priority_colors = {
            AlertPriority.INFO: 0x00ff00,      # Green
            AlertPriority.WARNING: 0xffa500,   # Orange
            AlertPriority.CRITICAL: 0xff0000,  # Red
            AlertPriority.EMERGENCY: 0x8b0000  # Dark Red
        }
        
        color = priority_colors.get(alert.priority, 0x00ff00)
        
        embed = {
            'title': alert.title,
            'description': alert.message,
            'color': color,
            'timestamp': alert.timestamp.isoformat(),
            'fields': [
                {
                    'name': 'Priority',
                    'value': alert.priority.value.upper(),
                    'inline': True
                },
                {
                    'name': 'Alert ID',
                    'value': alert.alert_id,
                    'inline': True
                }
            ],
            'footer': {
                'text': 'Smart Ape Mode Alert System'
            }
        }
        
        # Add data fields
        if alert.data:
            for key, value in list(alert.data.items())[:5]:  # Limit to 5 fields
                embed['fields'].append({
                    'name': key.replace('_', ' ').title(),
                    'value': str(value),
                    'inline': True
                })
        
        return embed
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format HTML email body"""
        
        priority_colors = {
            AlertPriority.INFO: "#28a745",
            AlertPriority.WARNING: "#ffc107", 
            AlertPriority.CRITICAL: "#dc3545",
            AlertPriority.EMERGENCY: "#8b0000"
        }
        
        color = priority_colors.get(alert.priority, "#28a745")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border-left: 4px solid {color}; padding-left: 20px;">
                <h2 style="color: {color}; margin-top: 0;">{alert.title}</h2>
                <p style="font-size: 16px; line-height: 1.5;">{alert.message}</p>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #495057;">Alert Details</h3>
                    <p><strong>Priority:</strong> {alert.priority.value.upper()}</p>
                    <p><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Alert ID:</strong> {alert.alert_id}</p>
                </div>
        """
        
        # Add data table if present
        if alert.data:
            html += """
                <div style="background-color: #e9ecef; padding: 15px; border-radius: 5px;">
                    <h3 style="margin-top: 0; color: #495057;">Additional Data</h3>
                    <table style="width: 100%; border-collapse: collapse;">
            """
            
            for key, value in alert.data.items():
                html += f"""
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6; font-weight: bold;">
                                {key.replace('_', ' ').title()}
                            </td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">
                                {value}
                            </td>
                        </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        html += """
            </div>
            <hr style="margin: 30px 0; border: none; border-top: 1px solid #dee2e6;">
            <p style="color: #6c757d; font-size: 12px;">
                This alert was generated by the Smart Ape Mode Alert System.
            </p>
        </body>
        </html>
        """
        
        return html
    
    async def send_system_status(self, status_data: Dict[str, Any]):
        """Send system status update"""
        
        await self.send_templated_alert(
            'system_health',
            priority=AlertPriority.INFO,
            channels=[AlertChannel.TELEGRAM],
            **status_data
        )
    
    async def send_profit_alert(self, profit: float, roi: float):
        """Send profit milestone alert"""
        
        await self.send_templated_alert(
            'profit_milestone',
            priority=AlertPriority.INFO,
            channels=[AlertChannel.ALL],
            profit=profit,
            roi=roi
        )
    
    async def send_emergency_alert(self, title: str, message: str, data: Dict[str, Any] = None):
        """Send emergency alert to all channels"""
        
        await self.send_alert(
            title=title,
            message=message,
            priority=AlertPriority.EMERGENCY,
            channels=[AlertChannel.ALL],
            data=data or {}
        )
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        
        return {
            'total_alerts_sent': len(self.alert_history),
            'alerts_in_queue': self.alert_queue.qsize(),
            'rate_limits': {
                channel.value: data['count']
                for channel, data in self.rate_limits.items()
            },
            'channels_configured': {
                'telegram': bool(self.telegram_bot_token and self.telegram_chat_id),
                'discord': bool(self.discord_webhook_url),
                'email': bool(self.email_user and self.alert_email)
            },
            'processing_active': self.processing_active
        }
    
    async def shutdown(self):
        """Graceful shutdown of alert system"""
        
        self.logger.info("📡 Shutting down Alert System")
        
        self.processing_active = False
        
        # Wait for queue to empty
        await self.alert_queue.join()
        
        self.logger.info("✅ Alert System shutdown complete")

def create_alert_system() -> AdvancedAlertSystem:
    """Create advanced alert system"""
    return AdvancedAlertSystem()
