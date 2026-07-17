import time
from typing import Generator, Dict, Any
from llm.base import LLMProvider

class MockProvider(LLMProvider):
    """
    Mock LLM provider for local testing and debugging.
    Generates realistic geotechnical landslide assessment reports without heavy hardware.
    """
    
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        time.sleep(1.0)  # Simulate API latency
        return self._get_mock_response(prompt)
        
    def generate_stream(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        response = self._get_mock_response(prompt)
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)  # Simulate token streaming
            
    def _get_mock_response(self, prompt: str) -> str:
        """Determines the style of prompt and returns appropriate structured text."""
        prompt_lower = prompt.lower()
        
        if "plain" in prompt_lower:
            return (
                "# Plain-Language Landslide Hazard Briefing\n\n"
                "## What is happening on the slope?\n"
                "Our satellite radar measurements show that a section of the slope named **Slope Sector Alpha-7** is slowly moving downwards. "
                "The movement speed is about **42.7 millimeters per year** on average, with some spots moving as fast as **118.4 millimeters per year**.\n\n"
                "## What does this mean for safety?\n"
                "The risk rating is currently classified as **Critical**. This means there is a high chance of a landslide happening, "
                "especially if the ground becomes heavily saturated with rain. Soil moisture is currently high, which lubricates the rock and makes sliding easier.\n\n"
                "## Warning signs to look out for:\n"
                "- New cracks appearing in the soil, roads, or building foundations.\n"
                "- Fences, utility poles, or trees tilting downslope.\n"
                "- Water leaking or bubbling up in new places on the hill.\n\n"
                "## Immediate advice:\n"
                "Please stay alert and follow the evacuation notices from local authorities."
            )
            
        elif "emergency" in prompt_lower or "recommendation" in prompt_lower:
            return (
                "# Civil Protection & Emergency Recommendations\n\n"
                "### 1. Immediate Actions (Next 24-48 Hours)\n"
                "- **Evacuation Zone Enforcement**: Restrict public access to the lower runout path of Sector Alpha-7.\n"
                "- **Road Closures**: Close secondary roads cutting across the active deformation boundary.\n"
                "- **Rainfall Monitoring**: Establish real-time telemetry threshold triggers (e.g. evacuation if rain exceeds 30mm/day).\n\n"
                "### 2. Short-Term Interventions (Next 1-4 Weeks)\n"
                "- **Geotechnical Sensor Grid**: Deploy automated wire extensometers and inclinometers across the head scarp.\n"
                "- **Temporary Drainage**: Dig plastic-lined channels to divert surface runoff away from tension cracks.\n"
                "- **Erosion Protection**: Install high-tensile steel mesh netting on steep exposures.\n\n"
                "### 3. Long-Term Stabilization\n"
                "- **Deep Drainage Wells**: Install sub-horizontal drill-hole drains to lower the water table within the slope body.\n"
                "- **Retaining Structures**: Construct a concrete pile retaining wall anchored into the limestone bedrock at the slope toe.\n"
                "- **Vegetative Soil Stabilization**: Plant deep-rooting grasses and shrubs to tie the topsoil layer together."
            )
            
        elif "question" in prompt_lower or "chat" in prompt_lower or "user" in prompt_lower:
            return (
                "Based on the landslide assessment report for Slope Sector Alpha-7, "
                "the maximum recorded movement velocity is 118.4 mm/yr. "
                "This speed, combined with a high probability of failure (89%) and high soil moisture, "
                "justifies the 'Critical' hazard risk level. Geotechnical teams recommend immediate "
                "evacuation boundaries and surface water diversion channel construction."
            )
            
        else:
            # Default to Professional Technical Report
            return (
                "# Landslide Geohazard Assessment Report\n"
                "**Report ID:** LSA-2026-07-06  \n"
                "**Site Name:** Slope Sector Alpha-7  \n"
                "**Assessment Date:** 2026-07-06  \n"
                "**Geographic Center:** Lat 45.8912°, Lon 7.2415°  \n"
                "**Risk Category:** Critical\n\n"
                "---\n\n"
                "## 1. Executive Summary\n"
                "This report presents a satellite-radar deformation assessment of Slope Sector Alpha-7 using Multi-Temporal InSAR. "
                "A highly active deformation zone covering 12,450 sq meters has been detected. Based on structural indicators, "
                "the risk of major slope failure is critical, warranting immediate civil protection response.\n\n"
                "## 2. Multi-Temporal InSAR Displacement Analysis\n"
                "Radar interferometry shows active downslope creep. The mean Line-of-Sight (LOS) velocity is **-42.7 mm/yr**, "
                "with localized peak velocity reaching **-118.4 mm/yr**. Negative velocities reflect movements away from the satellite, "
                "consistent with downslope mass sliding. Temporal profiles indicate a non-linear progressive acceleration trend, "
                "suggesting the slope has entered a tertiary creep phase.\n\n"
                "## 3. Risk Assessment & Geological Context\n"
                "The geological units consist of highly fractured limestone overlying weak shale layers. Aspect is North-West with "
                "a steep 38.5° slope gradient. Current soil moisture is high, acting as the primary pore-water pressure trigger. "
                "Under current conditions, the computed probability of slope failure is **0.89** with a hazard severity score of **8.4/10**.\n\n"
                "## 4. Engineering & Safety Recommendations\n"
                "1. Establish an immediate exclusionary safety perimeter at the toe of the slope.\n"
                "2. Execute surface drainage re-routing to minimize infiltration into the upper tension cracks.\n"
                "3. Install a continuous real-time GNSS monitoring grid paired with automated warning alerts."
            )
