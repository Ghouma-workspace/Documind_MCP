"""
Automation Agent - Handles template filling and file operations
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from jinja2 import Template, Environment, FileSystemLoader, TemplateNotFound

from agents.base_agent import BaseAgent
from aop.protocol import (
    AOPMessage,
    AOPProtocol,
    AgentType,
    MessageType,
    AutomationRequest,
    AutomationResponse
)

logger = logging.getLogger(__name__)


class AutomationAgent(BaseAgent):
    """
    Automation Agent - File operations and template management
    - Fills document templates
    - Saves generated outputs
    - Exports results in various formats
    - Manages file operations
    """
    
    def __init__(
        self,
        protocol: AOPProtocol,
        templates_path: str,
        outputs_path: str
    ):
        super().__init__(AgentType.AUTOMATION, "AutomationAgent")
        self.protocol = protocol
        self.templates_path = Path(templates_path)
        self.outputs_path = Path(outputs_path)
        self.automation_history = []
        
        # Ensure directories exist
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=True
        )
    
    async def process(self, message: AOPMessage) -> AOPMessage:
        """Process incoming automation request"""
        self.log_info(f"Processing message: {message.message_type}")
        
        try:
            if message.message_type == MessageType.AUTOMATION_REQUEST:
                return await self._handle_automation_request(message)
            else:
                return self._create_error_response(
                    message,
                    "Unsupported message type",
                    f"Cannot process message type: {message.message_type}"
                )
        except Exception as e:
            self.log_error(f"Error processing message: {str(e)}", exc_info=True)
            return self._create_error_response(message, "ProcessingError", str(e))
    
    async def _handle_automation_request(self, message: AOPMessage) -> AOPMessage:
        """Handle automation request"""
        try:
            # Parse request
            auto_req = AutomationRequest(**message.payload)
            
            self.log_info(f"Executing automation action: {auto_req.action}")
            
            # Execute action
            if auto_req.action == "fill_template":
                result = await self._fill_template(auto_req)
            elif auto_req.action == "save_file":
                result = await self._save_file(auto_req)
            elif auto_req.action == "export":
                result = await self._export_file(auto_req)
            else:
                raise ValueError(f"Unknown action: {auto_req.action}")
            
            # Store in history
            self.automation_history.append({
                "action": auto_req.action,
                "timestamp": datetime.utcnow().isoformat(),
                "output_path": result.get("output_path"),
                "success": result.get("success", False)
            })
            
            # Create response
            auto_resp = AutomationResponse(
                action=auto_req.action,
                success=result["success"],
                output_path=result.get("output_path"),
                message=result.get("message", "Action completed")
            )
            
            return self.protocol.create_message(
                message_type=MessageType.AUTOMATION_RESPONSE,
                sender=self.agent_type,
                receiver=message.sender,
                payload=auto_resp.model_dump(),
                parent_message_id=message.message_id
            )
            
        except Exception as e:
            self.log_error(f"Error handling automation request: {str(e)}", exc_info=True)
            return self._create_error_response(message, "AutomationError", str(e))
    
    async def _fill_template(self, request: AutomationRequest) -> Dict[str, Any]:
        """Fill a template with data"""
        try:
            template_name = request.template_name
            if not template_name:
                raise ValueError("Template name is required")
            
            # Add extension if not present
            if not template_name.endswith(('.md', '.txt', '.html', '.json')):
                # Try common extensions first
                possible_extensions = ['.md', '.txt', '.html', '.json']
                template_found = False
                
                for ext in possible_extensions:
                    test_name = template_name + ext
                    try:
                        self.jinja_env.get_template(test_name)
                        template_name = test_name
                        template_found = True
                        break
                    except TemplateNotFound:
                        continue
                
                # If not found with common extensions, use the requested output format
                if not template_found:
                    if request.output_format == "markdown":
                        template_name += ".md"
                    else:
                        template_name += f".{request.output_format}"
            
            self.log_info(f"Filling template: {template_name}")
            
            # Try to load template
            try:
                template = self.jinja_env.get_template(template_name)
            except TemplateNotFound:
                # Create default template if not found
                self.log_warning(f"Template {template_name} not found, using default")
                template = self._get_default_template(request.output_format)
            
            # Render template
            rendered = template.render(**request.data)
            
            # Generate output filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{template_name.split('.')[0]}_{timestamp}.{request.output_format}"
            output_path = self.outputs_path / output_filename
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rendered)
            
            self.log_info(f"Template filled and saved to: {output_path}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "message": f"Template filled successfully: {template_name}"
            }
            
        except Exception as e:
            self.log_error(f"Error filling template: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to fill template: {str(e)}"
            }
    
    async def _save_file(self, request: AutomationRequest) -> Dict[str, Any]:
        """Save data to file"""
        try:
            # Get content to save
            content = request.data.get("content", "")
            filename = request.data.get("filename")
            
            if not filename:
                # Generate filename
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"output_{timestamp}.{request.output_format}"
            
            output_path = self.outputs_path / filename
            
            # Save based on format
            if request.output_format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(request.data, f, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            
            self.log_info(f"File saved to: {output_path}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "message": f"File saved successfully: {filename}"
            }
            
        except Exception as e:
            self.log_error(f"Error saving file: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save file: {str(e)}"
            }
    
    async def _export_file(self, request: AutomationRequest) -> Dict[str, Any]:
        """Export data in specified format"""
        try:
            data = request.data
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}.{request.output_format}"
            
            if request.output_path:
                output_path = Path(request.output_path)
            else:
                output_path = self.outputs_path / filename
            
            # Export based on format
            if request.output_format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            elif request.output_format == "markdown":
                content = self._dict_to_markdown(data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif request.output_format == "txt":
                content = self._dict_to_text(data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            else:
                # Default: write as string
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            self.log_info(f"Data exported to: {output_path}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "message": f"Data exported successfully to {request.output_format}"
            }
            
        except Exception as e:
            self.log_error(f"Error exporting file: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to export file: {str(e)}"
            }
    
    def _get_default_template(self, output_format: str) -> Template:
        """Get default template for format"""
        if output_format == "markdown":
            template_str = """# {{ title | default("Document") }}

**Generated**: {{ date | default("N/A") }}

## Content

{{ content | default("No content provided") }}

---

{% if metadata %}
## Metadata
{% for key, value in metadata.items() %}
- **{{ key }}**: {{ value }}
{% endfor %}
{% endif %}
"""
        elif output_format == "json":
            template_str = """{{ data | tojson(indent=2) }}"""
        
        else:  # txt
            template_str = """{{ title | default("Document") }}
{{ "=" * 50 }}

{{ content | default("No content provided") }}

{% if metadata %}
Metadata:
{% for key, value in metadata.items() %}
{{ key }}: {{ value }}
{% endfor %}
{% endif %}
"""
        
        return Template(template_str)
    
    def _dict_to_markdown(self, data: Dict[str, Any], level: int = 1) -> str:
        """Convert dictionary to markdown format"""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key}\n")
                lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"{'#' * level} {key}\n")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_markdown(item, level + 1))
                    else:
                        lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(f"**{key}**: {value}\n")
        
        return "\n".join(lines)
    
    def _dict_to_text(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to plain text format"""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_text(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        
        return "\n".join(lines)
    
    async def fill_and_save(self, template_name: str, data: Dict[str, Any], output_format: str = "txt") -> Dict[str, Any]:
        """
        Direct template filling method (no MCP) - for API endpoints
        Returns: Dict with success, output_path, message
        """
        from aop.protocol import AutomationRequest
        
        self.log_info(f"Direct fill_and_save: {template_name}")
        
        # Create a request object
        request = AutomationRequest(
            action="fill_template",
            template_name=template_name,
            data=data,
            output_format=output_format
        )
        
        # Call internal method
        result = await self._fill_template(request)
        return result

    
    def create_template(self, name: str, content: str) -> bool:
        """Create a new template"""
        try:
            template_path = self.templates_path / name
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.log_info(f"Template created: {name}")
            return True
        except Exception as e:
            self.log_error(f"Error creating template: {str(e)}")
            return False
    
    def list_templates(self) -> list:
        """List available templates"""
        templates = []
        for template_file in self.templates_path.glob('*'):
            if template_file.is_file():
                templates.append(template_file.name)
        return templates
    
    def list_outputs(self) -> list:
        """List generated output files"""
        outputs = []
        for output_file in self.outputs_path.glob('*'):
            if output_file.is_file():
                outputs.append({
                    "filename": output_file.name,
                    "path": str(output_file),
                    "size": output_file.stat().st_size,
                    "modified": datetime.fromtimestamp(output_file.stat().st_mtime).isoformat()
                })
        return outputs
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return {
            "total_automations": len(self.automation_history),
            "templates_available": len(self.list_templates()),
            "outputs_generated": len(self.list_outputs()),
            "recent_actions": [h["action"] for h in self.automation_history[-10:]]
        }
    
    def _create_error_response(self, original_message: AOPMessage, error_type: str, error_message: str) -> AOPMessage:
        """Create error response"""
        return self.protocol.create_error_response(
            sender=self.agent_type,
            receiver=original_message.sender,
            error_type=error_type,
            error_message=error_message,
            parent_message_id=original_message.message_id,
            recoverable=True
        )
