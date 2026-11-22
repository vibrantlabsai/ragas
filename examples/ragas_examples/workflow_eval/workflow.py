import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional

from openai import OpenAI


@dataclass
class TraceEvent:
    """Single event in the application trace"""

    event_type: str  # "llm_call", "llm_response", "extraction", "classification", "error", "init"
    component: (
        str  # "openai_api", "deterministic_extractor", "llm_extractor", "support_agent"
    )
    data: Dict[str, Any]


class ExtractionMode(Enum):
    """Extraction modes available"""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


class BaseExtractor(ABC):
    """Base class for all extractors"""

    @abstractmethod
    def extract(self, email_content: str, category: str) -> Dict[str, Optional[str]]:
        """Extract information based on category"""
        pass


class DeterministicExtractor(BaseExtractor):
    """Regex and rule-based extraction"""

    def extract(self, email_content: str, category: str) -> Dict[str, Optional[str]]:
        """Route to appropriate extraction method"""
        extractors = {
            "Bug Report": self._extract_bug_info,
            "Billing": self._extract_billing_info,
            "Feature Request": self._extract_feature_info,
        }

        extractor = extractors.get(category)
        if extractor:
            return extractor(email_content)
        return {}

    def _extract_bug_info(self, email_content: str) -> Dict[str, Optional[str]]:
        """Extract product version and error code from bug reports"""
        version_pattern = r"version\s*[:\-]?\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)"
        error_pattern = r"error\s*(?:code\s*)?[:\-]?\s*([A-Z0-9\-_]+)"

        version_match = re.search(version_pattern, email_content, re.IGNORECASE)
        error_match = re.search(error_pattern, email_content, re.IGNORECASE)

        return {
            "product_version": version_match.group(1) if version_match else None,
            "error_code": error_match.group(1) if error_match else None,
        }

    def _extract_billing_info(self, email_content: str) -> Dict[str, Optional[str]]:
        """Extract invoice number and amount from billing emails"""
        invoice_pattern = r"invoice\s*[#:\-]?\s*([A-Z0-9\-_]+)"
        amount_pattern = r"\$([0-9,]+(?:\.[0-9]{2})?)"

        invoice_match = re.search(invoice_pattern, email_content, re.IGNORECASE)
        amount_match = re.search(amount_pattern, email_content)

        # Clean up amount (remove commas)
        amount = None
        if amount_match:
            amount = amount_match.group(1).replace(",", "")

        return {
            "invoice_number": invoice_match.group(1) if invoice_match else None,
            "amount": amount,
        }

    def _extract_feature_info(self, email_content: str) -> Dict[str, Optional[str]]:
        """Extract feature request details"""
        # Urgency detection
        urgency_keywords = {
            "urgent": ["urgent", "asap", "immediately", "critical", "emergency"],
            "high": ["important", "soon", "needed", "priority", "essential"],
            "medium": ["would like", "request", "suggest", "consider"],
            "low": ["nice to have", "whenever", "eventually", "someday"],
        }

        urgency_level = "medium"  # default
        email_lower = email_content.lower()

        for level, keywords in urgency_keywords.items():
            if any(keyword in email_lower for keyword in keywords):
                urgency_level = level
                break

        # Product area detection
        product_areas = [
            "dashboard",
            "api",
            "mobile",
            "reports",
            "billing",
            "user management",
            "analytics",
            "integration",
            "security",
        ]
        mentioned_areas = [area for area in product_areas if area in email_lower]

        # Try to extract the main feature request (simple approach)
        feature_keywords = [
            "add",
            "feature",
            "ability",
            "support",
            "implement",
            "create",
        ]
        requested_feature = None

        for keyword in feature_keywords:
            pattern = rf"{keyword}\s+(?:a\s+|an\s+|the\s+)?([^.!?]+)"
            match = re.search(pattern, email_content, re.IGNORECASE)
            if match:
                requested_feature = match.group(1).strip()[:100]  # Limit length
                break

        return {
            "requested_feature": requested_feature
            or "Feature extraction requires manual review",
            "product_area": mentioned_areas[0] if mentioned_areas else "general",
            "urgency_level": urgency_level,
        }


class LLMExtractor(BaseExtractor):
    """LLM-based extraction"""

    def __init__(self, client: OpenAI):
        self.client = client

    def extract(self, email_content: str, category: str) -> Dict[str, Optional[str]]:
        """Use LLM to extract information"""

        extraction_prompts = {
            "Bug Report": self._get_bug_extraction_prompt,
            "Billing": self._get_billing_extraction_prompt,
            "Feature Request": self._get_feature_extraction_prompt,
        }

        prompt_func = extraction_prompts.get(category)
        if not prompt_func:
            return {}

        prompt = prompt_func(email_content)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )

            # Parse JSON response
            result = json.loads(
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else "{}"
            )
            return result

        except Exception:
            return {}

    def _get_bug_extraction_prompt(self, email_content: str) -> str:
        return f"""
        Extract the following information from this bug report email:
        - product_version: The version number mentioned (e.g., "2.1.4")
        - error_code: Any error code mentioned (e.g., "XYZ-123")
        
        Email: {email_content}
        
        Respond with valid JSON only, like:
        {{"product_version": "2.1.4", "error_code": "XYZ-123"}}
        
        If a field is not found, use null.
        """

    def _get_billing_extraction_prompt(self, email_content: str) -> str:
        return f"""
        Extract the following information from this billing email:
        - invoice_number: The invoice number (e.g., "INV-2024-001")
        - amount: The dollar amount mentioned (without $ sign, e.g., "299.99")
        
        Email: {email_content}
        
        Respond with valid JSON only, like:
        {{"invoice_number": "INV-2024-001", "amount": "299.99"}}
        
        If a field is not found, use null.
        """

    def _get_feature_extraction_prompt(self, email_content: str) -> str:
        return f"""
        Extract the following information from this feature request email:
        - requested_feature: Brief description of the main feature requested (max 100 chars)
        - product_area: Which area it relates to (dashboard/api/mobile/reports/billing/user management/analytics/integration/security/general)
        - urgency_level: Urgency level (urgent/high/medium/low)
        
        Email: {email_content}
        
        Respond with valid JSON only, like:
        {{"requested_feature": "dark mode for dashboard", "product_area": "dashboard", "urgency_level": "high"}}
        
        If a field is not found, use appropriate defaults.
        """


class ConfigurableSupportTriageAgent:
    """Support triage agent with configurable extraction modes"""

    def __init__(
        self,
        api_key: str,
        extractor: Optional[BaseExtractor] = None,
        logdir: str = "logs",
    ):
        self.client = OpenAI(api_key=api_key)
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # If no extractor provided, default to deterministic
        if extractor is None:
            self.extractor = DeterministicExtractor()
        else:
            self.extractor = extractor

        # Store the extractor type for reference
        if isinstance(self.extractor, DeterministicExtractor):
            self.extraction_mode = ExtractionMode.DETERMINISTIC
        elif isinstance(self.extractor, LLMExtractor):
            self.extraction_mode = ExtractionMode.LLM
        else:
            # Custom extractor
            self.extraction_mode = None

        print(
            f"📧 Initialized Support Triage Agent with {self.extraction_mode.value if self.extraction_mode else 'custom'} extraction mode"
        )

        self.traces.append(
            TraceEvent(
                event_type="init",
                component="support_agent",
                data={
                    "extraction_mode": (
                        self.extraction_mode.value if self.extraction_mode else "custom"
                    )
                },
            )
        )

    def set_extractor(self, extractor: BaseExtractor):
        """Change extractor at runtime"""
        self.extractor = extractor

        # Update extraction mode
        if isinstance(self.extractor, DeterministicExtractor):
            self.extraction_mode = ExtractionMode.DETERMINISTIC
        elif isinstance(self.extractor, LLMExtractor):
            self.extraction_mode = ExtractionMode.LLM
        else:
            self.extraction_mode = None

        print(
            f"🔄 Switched to {self.extraction_mode.value if self.extraction_mode else 'custom'} extraction mode"
        )

        self.traces.append(
            TraceEvent(
                event_type="extractor_change",
                component="support_agent",
                data={
                    "new_extractor": type(extractor).__name__,
                    "extraction_mode": (
                        self.extraction_mode.value if self.extraction_mode else "custom"
                    ),
                },
            )
        )

    def classify_email(self, email_content: str) -> str:
        """Classify email into categories using LLM"""
        print("🔍 Step 1: Classifying email category...")

        prompt = f"""Classify the following customer email into exactly one of these categories:
        - Billing
        - Bug Report
        - Feature Request

        Email content:
        {email_content}

        Respond with only the category name, nothing else."""

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "classification",
                    "model": "gpt-3.5-turbo",
                    "prompt_length": len(prompt),
                    "email_length": len(email_content),
                },
            )
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            category = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else "unknown"
            )
            print(f"   ➜ Classified as: {category}")

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "classification",
                        "result": category,
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                    },
                )
            )

            return category

        except Exception as e:
            print("   ⚠️ Classification failed, using fallback: Bug Report")
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "classification", "error": str(e)},
                )
            )
            return "Bug Report"  # Default fallback

    def extract_info(
        self, email_content: str, category: str
    ) -> Dict[str, Optional[str]]:
        """Extract information using configured extractor"""
        print(
            f"⚙️  Step 2: Extracting {category} details using {self.extraction_mode.value if self.extraction_mode else 'custom'} method..."
        )

        self.traces.append(
            TraceEvent(
                event_type="extraction",
                component=type(self.extractor).__name__.lower(),
                data={
                    "category": category,
                    "email_length": len(email_content),
                    "extraction_mode": (
                        self.extraction_mode.value if self.extraction_mode else "custom"
                    ),
                },
            )
        )

        try:
            result = self.extractor.extract(email_content, category)

            # Show extracted fields briefly
            if result:
                extracted_fields = [k for k, v in result.items() if v is not None]
                if extracted_fields:
                    print(f"   ➜ Extracted: {', '.join(extracted_fields)}")
                else:
                    print("   ➜ No specific details extracted")

            self.traces.append(
                TraceEvent(
                    event_type="extraction_result",
                    component=type(self.extractor).__name__.lower(),
                    data={"extracted_fields": list(result.keys()), "result": result},
                )
            )

            return result

        except Exception as e:
            print(f"   ⚠️ Extraction failed: {str(e)}")
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component=type(self.extractor).__name__.lower(),
                    data={"operation": "extraction", "error": str(e)},
                )
            )
            return {}

    def generate_response(self, category: str, extracted_info: Dict[str, Any]) -> str:
        """Generate response template based on category"""
        print("✍️  Step 3: Generating personalized response...")

        context = f"Category: {category}\nExtracted info: {json.dumps(extracted_info, indent=2)}"

        prompt = f"""
        Generate a professional customer support response template for the following:
        
        {context}
        
        The response should:
        - Be polite and professional
        - Acknowledge the specific issue type
        - Include next steps or resolution process
        - Reference any extracted information appropriately
        
        Keep it concise but helpful.
        """

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "response_generation",
                    "model": "gpt-3.5-turbo",
                    "category": category,
                    "extracted_fields": list(extracted_info.keys()),
                },
            )
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )

            response_text = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )
            print("   ➜ Response template generated")

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "response_generation",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                    },
                )
            )

            return response_text

        except Exception as e:
            print("   ⚠️ Response generation failed, using fallback")
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "response_generation", "error": str(e)},
                )
            )
            return "Thank you for contacting support. We will review your request and get back to you soon."

    def export_traces_to_log(
        self, run_id: str, email_content: str, result: Optional[Dict[str, Any]] = None
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "email_content": email_content,
            "result": result,
            "extraction_mode": (
                self.extraction_mode.value if self.extraction_mode else "custom"
            ),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        return log_filepath

    def process_email(
        self, email_content: str, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main processing function that handles the entire workflow"""

        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(email_content) % 10000:04d}"

        print(f"\n🚀 Processing email (Run ID: {run_id})")
        print(
            f"📄 Email preview: {email_content[:100]}{'...' if len(email_content) > 100 else ''}"
        )

        # Reset traces for each new email
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="workflow_start",
                component="support_agent",
                data={"run_id": run_id, "email_length": len(email_content)},
            )
        )

        try:
            # Step 1: Classify email
            category = self.classify_email(email_content)

            # Step 2: Extract relevant information based on category
            extracted_info = self.extract_info(email_content, category)

            # Step 3: Generate response template
            response_template = self.generate_response(category, extracted_info)

            result = {
                "category": category,
                "extracted_info": extracted_info,
                "response_template": response_template,
                "extraction_mode": (
                    self.extraction_mode.value if self.extraction_mode else "custom"
                ),
            }

            print("✅ Workflow completed successfully")
            print(f"📋 Traces saved to: logs/run_{run_id}_*.json")

            self.traces.append(
                TraceEvent(
                    event_type="workflow_complete",
                    component="support_agent",
                    data={"run_id": run_id, "success": True},
                )
            )

            # Export traces to log file
            self.export_traces_to_log(run_id, email_content, result)

            return result

        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")

            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="support_agent",
                    data={"operation": "process_email", "error": str(e)},
                )
            )

            # Export traces even if processing failed
            self.export_traces_to_log(run_id, email_content, {})

            # Return minimal result on error
            return {
                "category": "Bug Report",
                "extracted_info": {},
                "response_template": "Thank you for contacting support. We will review your request and get back to you soon.",
                "extraction_mode": (
                    self.extraction_mode.value if self.extraction_mode else "custom"
                ),
            }


def default_workflow_client(
    extractor_type: Literal["deterministic", "llm"] = "deterministic",
) -> ConfigurableSupportTriageAgent:
    """Create a default workflow client with specified extractor type"""
    print(f"🔧 Creating workflow client with {extractor_type} extraction...")

    api_key = os.environ.get("OPENAI_API_KEY")

    if extractor_type == "deterministic":
        extractor = DeterministicExtractor()
    elif extractor_type == "llm":
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for LLM extractor"
            )
        client = OpenAI(api_key=api_key)
        extractor = LLMExtractor(client)
    else:
        raise ValueError(f"Unsupported extractor type: {extractor_type}")

    # Use a default API key if none provided and using deterministic extractor
    if api_key is None:
        api_key = "dummy"

    return ConfigurableSupportTriageAgent(
        api_key=api_key, extractor=extractor, logdir="logs"
    )


# Example usage and testing
def main():
    # Initialize the agent with different extractors
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        api_key = "dummy"

    # Test emails
    test_emails = [
        "Hi, I'm getting error code XYZ-123 when using version 2.1.4 of your software. Please help!",
        "I need to dispute invoice #INV-2024-001 for 299.99 dollars. The charge seems incorrect.",
    ]

    # Example 1: Using deterministic extractor
    print("\n=== Using Deterministic Extractor ===")
    deterministic_extractor = DeterministicExtractor()
    agent = ConfigurableSupportTriageAgent(
        api_key=api_key, extractor=deterministic_extractor, logdir="logs"
    )

    result = agent.process_email(test_emails[0])
    print(f"Result: {result['response_template']}")


if __name__ == "__main__":
    main()
