# alert_system.py
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import asyncio

class AlertSystem:
    """Handles 100-level alerts at drone coordinates"""
    
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.drone_coordinates = None
        self.alert_thresholds = {
            'human_detection': True,
            'emergency_signal': True
        }
    
    def update_drone_coordinates(self, lat: float, lon: float, altitude: float = None):
        """Update current drone position"""
        self.drone_coordinates = {
            'lat': lat,
            'lon': lon,
            'altitude': altitude,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_alert_conditions(self, human_detected: bool = False) -> bool:
        """Check if conditions warrant a 100-level alert"""
        conditions_met = []
        
        # Condition 1: Human detected at drone location
        if human_detected and self.alert_thresholds['human_detection']:
            conditions_met.append("Human detected")
        
        return len(conditions_met) > 0, conditions_met
    
    def create_alert(self, alert_type: str, conditions: List[str], severity: int = 100) -> Dict:
        """Create a new alert"""
        if not self.drone_coordinates:
            raise ValueError("Drone coordinates not set")
        
        alert = {
            'id': f"alert_{int(time.time() * 1000)}",
            'type': alert_type,
            'severity': severity,
            'coordinates': self.drone_coordinates.copy(),
            'conditions': conditions,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        return alert
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all currently active alerts"""
        return [alert for alert in self.active_alerts if alert['status'] == 'active']
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now().isoformat()
                break
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert system status"""
        active_count = len(self.get_active_alerts())
        total_count = len(self.alert_history)
        
        return {
            'active_alerts': active_count,
            'total_alerts': total_count,
            'drone_coordinates': self.drone_coordinates,
            'last_alert': self.alert_history[-1] if self.alert_history else None
        }

# Global alert system instance
alert_system = AlertSystem()

async def monitor_for_alerts():
    """Continuously monitor for alert conditions"""
    while True:
        try:
            # This would integrate with your existing human detection
            # For now, we'll simulate the monitoring
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in alert monitoring: {e}")
            await asyncio.sleep(5)

def trigger_100_level_alert(human_detected: bool = False) -> Optional[Dict]:
    """Main function to trigger 100-level alerts"""
    should_alert, conditions = alert_system.check_alert_conditions(human_detected)
    
    if should_alert:
        alert = alert_system.create_alert(
            alert_type="100_level_emergency",
            conditions=conditions,
            severity=100
        )
        print(f"100-LEVEL ALERT TRIGGERED: {', '.join(conditions)}")
        print(f"Location: {alert['coordinates']['lat']:.4f}, {alert['coordinates']['lon']:.4f}")
        return alert
    
    return None
