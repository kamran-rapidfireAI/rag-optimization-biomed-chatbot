"""
BioRAG Design Token Architecture

A single source of truth for all colors, typography, and spacing.
This module generates both CSS variables and Gradio theme from one definition.

Architecture:
    1. Primitive Tokens: Raw color values (e.g., blue_500 = "#58a6ff")
    2. Semantic Tokens: Purpose-based (e.g., text_primary → gray_100)
    3. Component Tokens: Component-specific (e.g., button_bg → accent_blue)

Usage:
    from demo.theme import BioRAGTheme
    theme = BioRAGTheme()
    css = theme.to_css()
    gradio_theme = theme.to_gradio_theme()
"""

from dataclasses import dataclass, field
from typing import Dict
import gradio as gr


# =============================================================================
# LAYER 1: PRIMITIVE TOKENS (Raw Design Values)
# =============================================================================

@dataclass(frozen=True)
class ColorPrimitives:
    """Raw color palette - the atomic building blocks."""
    
    # Grays (GitHub dark palette)
    gray_950: str = "#010409"   # Darkest
    gray_900: str = "#0d1117"   # Primary background
    gray_850: str = "#161b22"   # Secondary background
    gray_800: str = "#21262d"   # Tertiary/elevated
    gray_700: str = "#30363d"   # Borders
    gray_600: str = "#484f58"   # Subtle borders
    gray_500: str = "#6e7681"   # Disabled text
    gray_400: str = "#7d8590"   # Muted text
    gray_300: str = "#9198a1"   # Secondary text
    gray_200: str = "#b1bac4"   # 
    gray_100: str = "#e6edf3"   # Primary text
    white: str = "#ffffff"      # Brightest
    
    # Blues (Primary accent)
    blue_900: str = "#0c2d6b"
    blue_700: str = "#1158c7"
    blue_500: str = "#2f81f7"   # Standard
    blue_400: str = "#58a6ff"   # Bright (primary accent)
    blue_300: str = "#79c0ff"
    blue_200: str = "#a5d6ff"
    
    # Greens (Success/Optimized)
    green_700: str = "#238636"
    green_500: str = "#3fb950"  # Standard
    green_400: str = "#56d364"
    green_300: str = "#7ee787"
    
    # Reds (Error/Warning)
    red_700: str = "#da3633"
    red_500: str = "#f85149"   # Standard
    red_400: str = "#ff7b72"
    
    # Purples (Accent secondary)
    purple_500: str = "#8957e5"
    purple_400: str = "#a371f7"
    purple_300: str = "#bc8cff"
    
    # Oranges (Caution)
    orange_500: str = "#d29922"
    orange_400: str = "#e3b341"


# =============================================================================
# LAYER 2: SEMANTIC TOKENS (Purpose-Based)
# =============================================================================

@dataclass
class SemanticTokens:
    """Semantic tokens that map primitives to purposes."""
    
    primitives: ColorPrimitives = field(default_factory=ColorPrimitives)
    
    # --- Backgrounds ---
    @property
    def bg_app(self) -> str:
        """Main application background"""
        return self.primitives.gray_900
    
    @property
    def bg_surface(self) -> str:
        """Card/panel backgrounds"""
        return self.primitives.gray_850
    
    @property
    def bg_elevated(self) -> str:
        """Elevated elements (dropdowns, tooltips)"""
        return self.primitives.gray_800
    
    @property
    def bg_input(self) -> str:
        """Input field backgrounds"""
        return self.primitives.gray_800
    
    # --- Text ---
    @property
    def text_primary(self) -> str:
        """Main readable text"""
        return self.primitives.gray_100
    
    @property
    def text_secondary(self) -> str:
        """Less prominent text"""
        return self.primitives.gray_300
    
    @property
    def text_muted(self) -> str:
        """Placeholder, disabled text"""
        return self.primitives.gray_400
    
    @property
    def text_inverse(self) -> str:
        """Text on colored backgrounds"""
        return self.primitives.white
    
    # --- Borders ---
    @property
    def border_default(self) -> str:
        """Default border color"""
        return self.primitives.gray_700
    
    @property
    def border_subtle(self) -> str:
        """Subtle/faint borders"""
        return self.primitives.gray_600
    
    # --- Accents ---
    @property
    def accent_primary(self) -> str:
        """Primary interactive elements"""
        return self.primitives.blue_400
    
    @property
    def accent_primary_hover(self) -> str:
        """Primary hover state"""
        return self.primitives.blue_300
    
    @property
    def accent_success(self) -> str:
        """Success states"""
        return self.primitives.green_500
    
    @property
    def accent_error(self) -> str:
        """Error states"""
        return self.primitives.red_500
    
    @property
    def accent_warning(self) -> str:
        """Warning states"""
        return self.primitives.orange_500
    
    @property
    def accent_secondary(self) -> str:
        """Secondary accent (hover links, etc.)"""
        return self.primitives.purple_400


# =============================================================================
# LAYER 3: COMPONENT TOKENS (Component-Specific)
# =============================================================================

@dataclass
class ComponentTokens:
    """Component-specific tokens built on semantic tokens."""
    
    semantic: SemanticTokens = field(default_factory=SemanticTokens)
    
    # --- Buttons ---
    @property
    def button_primary_bg(self) -> str:
        return self.semantic.accent_primary
    
    @property
    def button_primary_bg_hover(self) -> str:
        return self.semantic.primitives.blue_500
    
    @property
    def button_primary_text(self) -> str:
        return self.semantic.text_inverse
    
    @property
    def button_secondary_bg(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def button_secondary_text(self) -> str:
        return self.semantic.text_primary
    
    # --- Inputs ---
    @property
    def input_bg(self) -> str:
        return self.semantic.bg_input
    
    @property
    def input_border(self) -> str:
        return self.semantic.border_default
    
    @property
    def input_text(self) -> str:
        return self.semantic.text_primary
    
    @property
    def input_placeholder(self) -> str:
        return self.semantic.text_muted
    
    # --- Dropdowns ---
    @property
    def dropdown_bg(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def dropdown_text(self) -> str:
        return self.semantic.text_primary
    
    @property
    def dropdown_option_bg(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def dropdown_option_bg_hover(self) -> str:
        return self.semantic.primitives.gray_700
    
    @property
    def dropdown_option_text(self) -> str:
        return self.semantic.text_primary
    
    # --- Tables ---
    @property
    def table_header_bg(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def table_cell_bg(self) -> str:
        return self.semantic.bg_surface
    
    @property
    def table_text(self) -> str:
        return self.semantic.text_primary
    
    @property
    def table_border(self) -> str:
        return self.semantic.border_default
    
    # --- Code ---
    @property
    def code_bg(self) -> str:
        return f"rgba(88, 166, 255, 0.15)"  # Blue tint
    
    @property
    def code_text(self) -> str:
        return self.semantic.accent_primary
    
    # --- Code Blocks (multi-line) ---
    @property
    def codeblock_bg(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def codeblock_text(self) -> str:
        return self.semantic.text_primary  # High contrast light text
    
    # --- Accordions ---
    @property
    def accordion_header_text(self) -> str:
        return self.semantic.text_secondary
    
    @property
    def accordion_content_bg(self) -> str:
        return self.semantic.bg_surface
    
    # --- Tabs ---
    @property
    def tab_text(self) -> str:
        return self.semantic.text_secondary
    
    @property
    def tab_text_hover(self) -> str:
        return self.semantic.text_primary
    
    @property
    def tab_bg_hover(self) -> str:
        return self.semantic.bg_elevated
    
    @property
    def tab_text_active(self) -> str:
        return self.semantic.accent_primary
    
    @property
    def tab_bg_active(self) -> str:
        return self.semantic.bg_surface
    
    # --- Pipeline Headers ---
    @property
    def pipeline_baseline_accent(self) -> str:
        return self.semantic.accent_primary  # Blue
    
    @property
    def pipeline_optimized_accent(self) -> str:
        return self.semantic.accent_success  # Green


# =============================================================================
# THEME GENERATOR
# =============================================================================

@dataclass
class BioRAGTheme:
    """
    Main theme class that generates CSS and Gradio theme from tokens.
    
    Single source of truth - all styling derives from here.
    """
    
    primitives: ColorPrimitives = field(default_factory=ColorPrimitives)
    semantic: SemanticTokens = field(default_factory=SemanticTokens)
    components: ComponentTokens = field(default_factory=ComponentTokens)
    
    # Typography
    font_family: str = "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif"
    font_family_mono: str = "'IBM Plex Mono', 'Consolas', monospace"
    
    # Spacing (rem-based)
    spacing_xs: str = "0.25rem"   # 4px
    spacing_sm: str = "0.5rem"    # 8px
    spacing_md: str = "1rem"      # 16px
    spacing_lg: str = "1.5rem"    # 24px
    spacing_xl: str = "2rem"      # 32px
    
    # Border radius
    radius_sm: str = "4px"
    radius_md: str = "8px"
    radius_lg: str = "12px"
    
    def to_css_variables(self) -> str:
        """Generate CSS custom properties from tokens."""
        return f"""
        :root {{
            /* === PRIMITIVES === */
            --color-gray-900: {self.primitives.gray_900};
            --color-gray-850: {self.primitives.gray_850};
            --color-gray-800: {self.primitives.gray_800};
            --color-gray-700: {self.primitives.gray_700};
            --color-gray-400: {self.primitives.gray_400};
            --color-gray-300: {self.primitives.gray_300};
            --color-gray-100: {self.primitives.gray_100};
            --color-white: {self.primitives.white};
            --color-blue-400: {self.primitives.blue_400};
            --color-blue-300: {self.primitives.blue_300};
            --color-green-500: {self.primitives.green_500};
            --color-red-500: {self.primitives.red_500};
            --color-purple-400: {self.primitives.purple_400};
            
            /* === SEMANTIC === */
            --bg-app: {self.semantic.bg_app};
            --bg-surface: {self.semantic.bg_surface};
            --bg-elevated: {self.semantic.bg_elevated};
            --bg-input: {self.semantic.bg_input};
            
            --text-primary: {self.semantic.text_primary};
            --text-secondary: {self.semantic.text_secondary};
            --text-muted: {self.semantic.text_muted};
            --text-inverse: {self.semantic.text_inverse};
            
            --border-default: {self.semantic.border_default};
            --border-subtle: {self.semantic.border_subtle};
            
            --accent-primary: {self.semantic.accent_primary};
            --accent-primary-hover: {self.semantic.accent_primary_hover};
            --accent-success: {self.semantic.accent_success};
            --accent-error: {self.semantic.accent_error};
            --accent-secondary: {self.semantic.accent_secondary};
            
            /* === COMPONENTS === */
            --button-primary-bg: {self.components.button_primary_bg};
            --button-primary-text: {self.components.button_primary_text};
            --input-bg: {self.components.input_bg};
            --input-border: {self.components.input_border};
            --input-text: {self.components.input_text};
            --input-placeholder: {self.components.input_placeholder};
            --dropdown-bg: {self.components.dropdown_bg};
            --dropdown-text: {self.components.dropdown_text};
            --dropdown-option-bg: {self.components.dropdown_option_bg};
            --dropdown-option-bg-hover: {self.components.dropdown_option_bg_hover};
            --dropdown-option-text: {self.components.dropdown_option_text};
            --table-header-bg: {self.components.table_header_bg};
            --table-cell-bg: {self.components.table_cell_bg};
            --table-text: {self.components.table_text};
            --code-bg: {self.components.code_bg};
            --code-text: {self.components.code_text};
            --codeblock-bg: {self.components.codeblock_bg};
            --codeblock-text: {self.components.codeblock_text};
            --tab-text: {self.components.tab_text};
            --tab-text-hover: {self.components.tab_text_hover};
            --tab-bg-hover: {self.components.tab_bg_hover};
            --tab-text-active: {self.components.tab_text_active};
            --tab-bg-active: {self.components.tab_bg_active};
            
            /* === TYPOGRAPHY === */
            --font-family: {self.font_family};
            --font-family-mono: {self.font_family_mono};
            
            /* === SPACING === */
            --spacing-xs: {self.spacing_xs};
            --spacing-sm: {self.spacing_sm};
            --spacing-md: {self.spacing_md};
            --spacing-lg: {self.spacing_lg};
            --spacing-xl: {self.spacing_xl};
            
            /* === RADIUS === */
            --radius-sm: {self.radius_sm};
            --radius-md: {self.radius_md};
            --radius-lg: {self.radius_lg};
        }}
        """
    
    def to_component_css(self) -> str:
        """Generate component-specific CSS using tokens."""
        return """
        /* ===========================================
           COMPONENT STYLES (using semantic tokens)
           =========================================== */
        
        /* --- Base Reset --- */
        .gradio-container {
            font-family: var(--font-family);
            background: var(--bg-app);
            color: var(--text-primary);
        }
        
        /* --- Global Text --- */
        .gradio-container,
        .gradio-container * {
            color: inherit;
        }
        
        /* --- Inputs --- */
        input, textarea, select,
        .gr-textbox input,
        .gr-textbox textarea {
            background-color: var(--input-bg) !important;
            border-color: var(--input-border) !important;
            color: var(--input-text) !important;
        }
        
        input::placeholder,
        textarea::placeholder {
            color: var(--input-placeholder);
        }
        
        /* --- Dropdowns --- */
        select,
        .gr-dropdown,
        [data-testid="dropdown"],
        .svelte-select,
        .choices,
        .choices__inner,
        .choices__list--dropdown {
            background-color: var(--dropdown-bg) !important;
            color: var(--dropdown-text) !important;
            border-color: var(--input-border) !important;
        }
        
        select option,
        .gr-dropdown option,
        .choices__item,
        .choices__list--dropdown .choices__item,
        [data-testid="dropdown"] li,
        .svelte-select .item,
        ul[role="listbox"] li,
        div[role="listbox"] > div,
        div[role="option"] {
            background-color: var(--dropdown-option-bg) !important;
            color: var(--dropdown-option-text) !important;
            padding: 8px 12px !important;
        }
        
        select option:hover,
        select option:focus,
        .gr-dropdown option:hover,
        .choices__item--selectable.is-highlighted,
        .choices__list--dropdown .choices__item:hover,
        [data-testid="dropdown"] li:hover,
        ul[role="listbox"] li:hover,
        ul[role="listbox"] li:focus,
        ul[role="listbox"] li[aria-selected="true"],
        div[role="option"]:hover,
        div[role="option"]:focus,
        div[role="option"][aria-selected="true"] {
            background-color: var(--dropdown-option-bg-hover) !important;
            color: var(--text-inverse) !important;
        }
        
        /* Dropdown arrow/icon fix */
        .gr-dropdown svg,
        select + svg,
        .choices::after {
            color: var(--text-secondary) !important;
        }
        
        /* --- Buttons --- */
        .gr-button-primary {
            background: linear-gradient(135deg, var(--button-primary-bg), var(--color-blue-300));
            color: var(--button-primary-text);
            border: none;
        }
        
        /* --- Tables --- */
        table, .gr-markdown table {
            background: var(--table-cell-bg);
            border-collapse: separate;
            border-spacing: 0;
        }
        
        th, .gr-markdown th {
            background: var(--table-header-bg);
            color: var(--text-primary);
            border-bottom: 2px solid var(--border-default);
            padding: 10px 12px;
        }
        
        td, .gr-markdown td {
            background: var(--table-cell-bg);
            color: var(--table-text);
            border-bottom: 1px solid var(--border-default);
            padding: 8px 12px;
        }
        
        /* --- Inline Code --- */
        code, .gr-markdown code {
            font-family: var(--font-family-mono);
            background: var(--code-bg);
            color: var(--code-text);
            padding: 2px 8px;
            border-radius: var(--radius-sm);
            font-size: 0.9em;
        }
        
        /* --- Code Blocks (multi-line) --- */
        pre, .gr-markdown pre {
            background: var(--codeblock-bg) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            padding: 16px !important;
            overflow-x: auto;
        }
        
        pre code, .gr-markdown pre code {
            background: transparent !important;
            color: var(--codeblock-text) !important;
            padding: 0 !important;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        /* --- Tabs --- */
        .tab-nav button,
        [role="tab"],
        button[role="tab"] {
            color: var(--tab-text) !important;
            background: transparent !important;
            transition: all 0.15s ease !important;
        }
        
        .tab-nav button:hover,
        [role="tab"]:hover,
        button[role="tab"]:hover {
            color: var(--tab-text-hover) !important;
            background: var(--tab-bg-hover) !important;
        }
        
        .tab-nav button.selected,
        [role="tab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            color: var(--tab-text-active) !important;
            background: var(--tab-bg-active) !important;
            border-bottom-color: var(--accent-primary) !important;
        }
        
        .tab-nav button.selected:hover,
        [role="tab"][aria-selected="true"]:hover,
        button[role="tab"][aria-selected="true"]:hover {
            color: var(--tab-text-active) !important;
            background: var(--tab-bg-active) !important;
        }
        
        /* --- Accordions --- */
        .gr-accordion > button,
        details > summary {
            color: var(--text-secondary);
        }
        
        .gr-accordion,
        .gr-accordion-content,
        details,
        details > div {
            background: var(--bg-surface);
            color: var(--text-primary);
        }
        
        /* --- Links --- */
        a, .gr-markdown a {
            color: var(--accent-primary);
        }
        
        a:hover {
            color: var(--accent-secondary);
        }
        
        /* --- Labels --- */
        label,
        .gr-input-label,
        .gr-block-label {
            color: var(--text-secondary);
        }
        
        /* --- Markdown --- */
        .gr-markdown,
        .gr-markdown p,
        .gr-markdown li {
            color: var(--text-primary);
        }
        
        .gr-markdown strong,
        .gr-markdown b {
            color: var(--text-inverse);
        }
        
        .gr-markdown em {
            color: var(--text-secondary);
        }
        
        .gr-markdown blockquote {
            border-left: 3px solid var(--accent-secondary);
            background: rgba(163, 113, 247, 0.08);
            padding: 12px 16px;
            color: var(--text-secondary);
        }
        """
    
    def to_layout_css(self) -> str:
        """Generate layout and decorative CSS."""
        return f"""
        /* ===========================================
           LAYOUT & DECORATIVE STYLES
           =========================================== */
        
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        
        .gradio-container {{
            background: 
                radial-gradient(ellipse at 20% 0%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(163, 113, 247, 0.08) 0%, transparent 50%),
                linear-gradient(180deg, var(--bg-app) 0%, var(--bg-surface) 100%);
            min-height: 100vh;
        }}
        
        /* --- Header --- */
        .main-header {{
            text-align: center;
            padding: 40px 24px;
            background: linear-gradient(180deg, rgba(88, 166, 255, 0.05) 0%, transparent 100%);
            border-bottom: 1px solid var(--border-default);
            margin-bottom: var(--spacing-xl);
        }}
        
        .main-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        }}
        
        .main-header h1 {{
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--text-inverse);
            margin: 0 0 8px 0;
            letter-spacing: -1px;
            text-shadow: 0 0 40px rgba(88, 166, 255, 0.3);
        }}
        
        .main-header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.15rem;
        }}
        
        .main-header .dna-icon {{
            font-size: 3rem;
            display: block;
            margin-bottom: 16px;
            filter: drop-shadow(0 0 20px rgba(88, 166, 255, 0.4));
        }}
        
        /* --- Pipeline Headers --- */
        .comparison-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 16px;
            border-radius: var(--radius-md);
            margin-bottom: 16px;
            font-weight: 600;
            font-size: 14px;
        }}
        
        .baseline-header {{
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.15) 0%, rgba(88, 166, 255, 0.05) 100%);
            border: 1px solid rgba(88, 166, 255, 0.3);
            color: var(--accent-primary);
        }}
        
        .optimized-header {{
            background: linear-gradient(135deg, rgba(63, 185, 80, 0.15) 0%, rgba(63, 185, 80, 0.05) 100%);
            border: 1px solid rgba(63, 185, 80, 0.3);
            color: var(--accent-success);
        }}
        
        /* --- Button Enhancements --- */
        .gr-button-primary {{
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(88, 166, 255, 0.25);
        }}
        
        .gr-button-primary:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(88, 166, 255, 0.4);
        }}
        
        /* --- Footer --- */
        .footer {{
            text-align: center;
            padding: var(--spacing-xl);
            margin-top: 48px;
            border-top: 1px solid var(--border-default);
            background: linear-gradient(180deg, transparent 0%, rgba(88, 166, 255, 0.02) 100%);
        }}
        
        .footer p {{
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .footer a {{
            color: var(--accent-primary);
            text-decoration: none;
        }}
        """
    
    def to_css(self) -> str:
        """Generate complete CSS from all token layers."""
        return (
            self.to_css_variables() + 
            self.to_component_css() + 
            self.to_layout_css()
        )
    
    def to_gradio_theme(self) -> gr.themes.Base:
        """Generate Gradio theme from tokens."""
        return gr.themes.Base(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=(self.font_family,),
            font_mono=(self.font_family_mono,),
        ).set(
            # Backgrounds
            body_background_fill=self.semantic.bg_app,
            body_background_fill_dark=self.semantic.bg_app,
            block_background_fill=self.semantic.bg_surface,
            block_background_fill_dark=self.semantic.bg_surface,
            
            # Borders
            block_border_color=self.semantic.border_default,
            
            # Text
            block_label_text_color=self.semantic.text_secondary,
            block_title_text_color=self.semantic.text_primary,
            body_text_color=self.semantic.text_primary,
            body_text_color_dark=self.semantic.text_primary,
            
            # Inputs
            input_background_fill=self.semantic.bg_input,
            input_background_fill_dark=self.semantic.bg_input,
            input_border_color=self.semantic.border_default,
            input_placeholder_color=self.semantic.text_muted,
            
            # Buttons
            button_primary_background_fill=self.components.button_primary_bg,
            button_primary_background_fill_hover=self.components.button_primary_bg_hover,
            button_primary_text_color=self.components.button_primary_text,
            button_secondary_background_fill=self.components.button_secondary_bg,
            button_secondary_text_color=self.components.button_secondary_text,
            
            # Tables
            table_text_color=self.components.table_text,
            table_text_color_dark=self.components.table_text,
            
            # Panels
            panel_background_fill=self.semantic.bg_surface,
            panel_background_fill_dark=self.semantic.bg_surface,
        )


# Singleton instance for easy import
theme = BioRAGTheme()





