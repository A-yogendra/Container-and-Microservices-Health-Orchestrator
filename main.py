import time
import random
from collections import deque
import json

# --- Configuration ---
# Define the services and their initial states
SERVICES_CONFIG = {
    "auth-service": {"cpu_usage": 0.3, "memory_usage": 0.4, "error_rate": 0.01, "latency": 50, "status": "Healthy"},
    "user-profile-service": {"cpu_usage": 0.2, "memory_usage": 0.3, "error_rate": 0.005, "latency": 40, "status": "Healthy"},
    "product-catalog-service": {"cpu_usage": 0.4, "memory_usage": 0.5, "error_rate": 0.02, "latency": 60, "status": "Healthy"},
    "order-processing-service": {"cpu_usage": 0.5, "memory_usage": 0.6, "error_rate": 0.03, "latency": 70, "status": "Healthy"},
}

# Thresholds for health assessment and failure detection
HEALTH_THRESHOLDS = {
    "cpu_critical": 0.85, "cpu_degraded": 0.70,
    "memory_critical": 0.80, "memory_degraded": 0.65,
    "error_critical": 0.05, "error_degraded": 0.02,
    "latency_critical": 150, "latency_degraded": 100,
}

# --- Global Variables for Simulation ---
# Store the current state of services
current_service_states = {name: dict(data) for name, data in SERVICES_CONFIG.items()}
# History for basic prediction (e.g., last N readings)
service_metrics_history = {name: deque(maxlen=10) for name in SERVICES_CONFIG}
# Store actions taken
action_log = []

# --- 1. Microservice Simulation ---
class Microservice:
    """
    Simulates a microservice with dynamic health metrics.
    In a real system, this data would come from actual monitoring agents.
    """
    def __init__(self, name, initial_metrics):
        self.name = name
        self.metrics = initial_metrics
        self.status = initial_metrics.get("status", "Healthy")
        print(f"[{self.name}] Initialized with status: {self.status}")

    def update_metrics(self):
        """
        Simulates metric fluctuations.
        Introduces random variations and occasional spikes to simulate issues.
        """
        # Introduce a small random fluctuation
        self.metrics["cpu_usage"] = max(0.01, min(0.99, self.metrics["cpu_usage"] + random.uniform(-0.05, 0.05)))
        self.metrics["memory_usage"] = max(0.01, min(0.99, self.metrics["memory_usage"] + random.uniform(-0.04, 0.04)))
        self.metrics["error_rate"] = max(0.001, min(0.1, self.metrics["error_rate"] + random.uniform(-0.005, 0.01)))
        self.metrics["latency"] = max(10, min(200, self.metrics["latency"] + random.uniform(-10, 20)))

        # Occasionally simulate a "problem"
        if random.random() < 0.1: # 10% chance of a spike
            problem_type = random.choice(["cpu", "memory", "error", "latency"])
            if problem_type == "cpu":
                self.metrics["cpu_usage"] = min(0.99, self.metrics["cpu_usage"] + random.uniform(0.1, 0.3))
            elif problem_type == "memory":
                self.metrics["memory_usage"] = min(0.99, self.metrics["memory_usage"] + random.uniform(0.1, 0.25))
            elif problem_type == "error":
                self.metrics["error_rate"] = min(0.1, self.metrics["error_rate"] + random.uniform(0.02, 0.05))
            elif problem_type == "latency":
                self.metrics["latency"] = min(200, self.metrics["latency"] + random.uniform(30, 80))

        # Update global state
        current_service_states[self.name].update(self.metrics)
        service_metrics_history[self.name].append(self.metrics.copy()) # Store a copy

    def __repr__(self):
        return f"Service({self.name}, Status: {self.status}, Metrics: {self.metrics})"

# --- 2. Health Monitoring Module ---
class HealthMonitor:
    """
    Monitors the health of simulated microservices based on thresholds.
    In a real system, this would analyze metrics from monitoring systems.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def assess_health(self, service_name, metrics):
        """
        Assesses the health of a single service based on its current metrics.
        Returns a status ('Healthy', 'Degraded', 'Critical') and a reason.
        """
        status = "Healthy"
        reasons = []

        if metrics["cpu_usage"] >= self.thresholds["cpu_critical"]:
            status = "Critical"
            reasons.append(f"High CPU ({metrics['cpu_usage']:.2f})")
        elif metrics["cpu_usage"] >= self.thresholds["cpu_degraded"]:
            if status == "Healthy": status = "Degraded"
            reasons.append(f"Elevated CPU ({metrics['cpu_usage']:.2f})")

        if metrics["memory_usage"] >= self.thresholds["memory_critical"]:
            status = "Critical"
            reasons.append(f"High Memory ({metrics['memory_usage']:.2f})")
        elif metrics["memory_usage"] >= self.thresholds["memory_degraded"]:
            if status == "Healthy": status = "Degraded"
            reasons.append(f"Elevated Memory ({metrics['memory_usage']:.2f})")

        if metrics["error_rate"] >= self.thresholds["error_critical"]:
            status = "Critical"
            reasons.append(f"High Error Rate ({metrics['error_rate']:.2%})")
        elif metrics["error_rate"] >= self.thresholds["error_degraded"]:
            if status == "Healthy": status = "Degraded"
            reasons.append(f"Elevated Error Rate ({metrics['error_rate']:.2%})")

        if metrics["latency"] >= self.thresholds["latency_critical"]:
            status = "Critical"
            reasons.append(f"High Latency ({metrics['latency']:.0f}ms)")
        elif metrics["latency"] >= self.thresholds["latency_degraded"]:
            if status == "Healthy": status = "Degraded"
            reasons.append(f"Elevated Latency ({metrics['latency']:.0f}ms)")

        return status, ", ".join(reasons) if reasons else "All metrics normal"

# --- 3. Failure Prediction Module (Simplified) ---
class FailurePredictor:
    """
    A very simplified "prediction" based on recent history.
    In a real system, this would involve ML models (e.g., time series analysis).
    """
    def predict_failure(self, service_name, history):
        """
        Checks if a metric has been consistently high or increasing.
        """
        if len(history) < 3: # Need at least 3 data points to see a trend
            return False, "Not enough history for prediction"

        # Check for increasing trend in CPU/Memory/Error Rate
        last_cpu = history[-1]["cpu_usage"]
        prev_cpu = history[-2]["cpu_usage"]
        prev_prev_cpu = history[-3]["cpu_usage"]

        last_mem = history[-1]["memory_usage"]
        prev_mem = history[-2]["memory_usage"]
        prev_prev_mem = history[-3]["memory_usage"]

        last_error = history[-1]["error_rate"]
        prev_error = history[-2]["error_rate"]
        prev_prev_error = history[-3]["error_rate"]

        if (last_cpu > prev_cpu > prev_prev_cpu and last_cpu > HEALTH_THRESHOLDS["cpu_degraded"]) or \
           (last_mem > prev_mem > prev_prev_mem and last_mem > HEALTH_THRESHOLDS["memory_degraded"]) or \
           (last_error > prev_error > prev_prev_error and last_error > HEALTH_THRESHOLDS["error_degraded"]):
            return True, "Metrics showing an increasing trend"

        return False, "No immediate prediction"

# --- 4. Failure Classification Module (Simplified) ---
class FailureClassifier:
    """
    A very simplified rule-based classifier.
    In a real system, this would be an ML model trained on failure patterns.
    """
    def classify_failure(self, service_name, status, reasons):
        """
        Classifies the type of failure based on status and reasons.
        """
        if status == "Critical":
            if "CPU" in reasons or "Memory" in reasons:
                return "RESOURCE_EXHAUSTION"
            elif "Error Rate" in reasons:
                return "APPLICATION_ERROR"
            elif "Latency" in reasons:
                # Could be network or application, for simplicity, we'll say performance
                return "PERFORMANCE_DEGRADATION"
        elif status == "Degraded":
            if "CPU" in reasons or "Memory" in reasons:
                return "RESOURCE_WARNING"
            elif "Error Rate" in reasons:
                return "APPLICATION_WARNING"
            elif "Latency" in reasons:
                return "PERFORMANCE_WARNING"
        return "UNKNOWN"

# --- 5. Healing Decision Engine ---
class HealingDecisionEngine:
    """
    Decides the optimal healing action based on service status, prediction, and classification.
    """
    def __init__(self, service_dependencies=None):
        # Example dependencies: order-processing depends on auth and product-catalog
        self.service_dependencies = service_dependencies or {
            "order-processing-service": ["auth-service", "product-catalog-service"],
            "user-profile-service": ["auth-service"]
        }

    def decide_action(self, service_name, current_status, predicted_failure, classified_type):
        """
        Determines the appropriate healing action.
        """
        print(f"  [Decision Engine] Analyzing {service_name}: Status={current_status}, Predicted={predicted_failure}, Classified={classified_type}")

        if current_status == "Healthy" and not predicted_failure:
            return None, "No action needed."

        action = None
        reason = "Default action"

        if classified_type == "RESOURCE_EXHAUSTION":
            action = "SCALE_UP"
            reason = "High resource usage detected, attempting to scale up."
        elif classified_type == "APPLICATION_ERROR":
            action = "RESTART_SERVICE"
            reason = "High error rate, attempting service restart."
        elif classified_type == "PERFORMANCE_DEGRADATION":
            action = "TRAFFIC_REROUTE" # Or scale up
            reason = "High latency, rerouting traffic to healthy instances (simulated)."
        elif predicted_failure:
            action = "SCALE_UP" # Proactive scaling
            reason = "Predicted future failure, proactively scaling up."
        elif current_status == "Degraded":
            action = "RESTART_SERVICE" # Mild issue, try restart
            reason = "Service degraded, attempting restart as a first step."

        # Consider dependencies (simplified logic)
        # In a real system, this would be much more sophisticated,
        # checking the health of dependencies before acting on the current service.
        if action and self.service_dependencies.get(service_name):
            for dep in self.service_dependencies[service_name]:
                dep_status = current_service_states.get(dep, {}).get("status")
                if dep_status == "Critical":
                    print(f"  [Decision Engine] WARNING: Dependency '{dep}' is Critical. Holding action on {service_name}.")
                    return None, f"Dependency '{dep}' is Critical. Holding action."

        return action, reason

# --- 6. Orchestration Module (Simulated) ---
class OrchestrationModule:
    """
    Simulates executing healing actions on the container orchestration platform.
    """
    def execute_action(self, service_name, action):
        """
        Simulates the execution of a given action.
        In a real system, this would call Kubernetes API, Docker API, etc.
        """
        if action == "RESTART_SERVICE":
            print(f"  [Orchestration] Executing: Restarting '{service_name}'...")
            # Simulate restart by resetting metrics to healthy levels
            current_service_states[service_name].update({
                "cpu_usage": random.uniform(0.1, 0.3),
                "memory_usage": random.uniform(0.1, 0.3),
                "error_rate": random.uniform(0.001, 0.005),
                "latency": random.uniform(20, 50),
                "status": "Healthy"
            })
            return True, f"Service '{service_name}' restarted."
        elif action == "SCALE_UP":
            print(f"  [Orchestration] Executing: Scaling up '{service_name}'...")
            # Simulate scaling by reducing current load
            current_service_states[service_name]["cpu_usage"] *= 0.7
            current_service_states[service_name]["memory_usage"] *= 0.7
            current_service_states[service_name]["status"] = "Healthy" # Assume scaling helps
            return True, f"Service '{service_name}' scaled up."
        elif action == "TRAFFIC_REROUTE":
            print(f"  [Orchestration] Executing: Rerouting traffic for '{service_name}'...")
            # Simulate rerouting by making service healthy again
            current_service_states[service_name].update({
                "cpu_usage": random.uniform(0.1, 0.3),
                "memory_usage": random.uniform(0.1, 0.3),
                "error_rate": random.uniform(0.001, 0.005),
                "latency": random.uniform(20, 50),
                "status": "Healthy"
            })
            return True, f"Traffic rerouted for '{service_name}'."
        else:
            print(f"  [Orchestration] Unknown action: {action} for {service_name}")
            return False, "Unknown action."

# --- Main Orchestrator Loop ---
def run_orchestrator(duration_seconds=60, tick_interval_seconds=2):
    """
    Main loop for the health orchestrator.
    """
    print("--- Starting Microservices Health Orchestrator ---")

    # Initialize components
    simulated_services = {name: Microservice(name, data) for name, data in SERVICES_CONFIG.items()}
    health_monitor = HealthMonitor(HEALTH_THRESHOLDS)
    failure_predictor = FailurePredictor()
    failure_classifier = FailureClassifier()
    decision_engine = HealingDecisionEngine()
    orchestrator = OrchestrationModule()

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < duration_seconds:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        for service_name, service_obj in simulated_services.items():
            print(f"\n[Monitoring] Processing service: {service_name}")

            # 1. Simulate Metrics Collection
            service_obj.update_metrics()
            current_metrics = current_service_states[service_name]
            print(f"  Current Metrics: CPU={current_metrics['cpu_usage']:.2f}, Mem={current_metrics['memory_usage']:.2f}, Err={current_metrics['error_rate']:.2%}, Latency={current_metrics['latency']:.0f}ms")

            # 2. Health Assessment
            status, reasons = health_monitor.assess_health(service_name, current_metrics)
            current_service_states[service_name]["status"] = status # Update global status
            print(f"  Health Status: {status} ({reasons})")

            # 3. Failure Prediction
            predicted, prediction_reason = failure_predictor.predict_failure(service_name, service_metrics_history[service_name])
            print(f"  Prediction: {'YES' if predicted else 'NO'} ({prediction_reason})")

            # 4. Failure Classification
            classified_type = failure_classifier.classify_failure(service_name, status, reasons)
            print(f"  Failure Classification: {classified_type}")

            # 5. Healing Decision Engine
            action, decision_reason = decision_engine.decide_action(service_name, status, predicted, classified_type)

            if action:
                print(f"  [Decision] Recommended Action: {action} ({decision_reason})")
                # 6. Orchestration Module
                success, execution_message = orchestrator.execute_action(service_name, action)
                print(f"  [Orchestration Result] {execution_message} (Success: {success})")
                action_log.append({
                    "timestamp": time.time(),
                    "iteration": iteration,
                    "service": service_name,
                    "action": action,
                    "status_before": status,
                    "metrics_before": current_metrics.copy(),
                    "success": success,
                    "message": execution_message
                })
            else:
                print(f"  [Decision] No action taken. {decision_reason}")

        time.sleep(tick_interval_seconds)

    print("\n--- Orchestrator Simulation Finished ---")
    print("\n--- Action Log ---")
    for entry in action_log:
        print(json.dumps(entry, indent=2))
    print(f"\nFinal Service States: {json.dumps(current_service_states, indent=2)}")

if __name__ == "__main__":
    run_orchestrator(duration_seconds=30, tick_interval_seconds=2) # Run for 30 seconds, check every 2 seconds
