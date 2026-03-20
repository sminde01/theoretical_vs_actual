import os
import streamlit.components.v1 as components

GA_ID = "G-8DM8073S27"

def load_ga(app_name: str) -> None:
    components.html(
        f"""
        <script>
        const host = window.parent.location.hostname;
        const isLocal =
            host === "localhost" ||
            host === "127.0.0.1" ||
            host === "0.0.0.0";

        if (!isLocal) {{
            const existingScript = window.parent.document.querySelector(
                'script[src*="googletagmanager.com/gtag/js?id={GA_ID}"]'
            );

            if (!existingScript) {{
                const s = window.parent.document.createElement("script");
                s.async = true;
                s.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
                window.parent.document.head.appendChild(s);
            }}

            window.parent.dataLayer = window.parent.dataLayer || [];
            function gtag() {{ window.parent.dataLayer.push(arguments); }}
            window.parent.gtag = window.parent.gtag || gtag;

            setTimeout(() => {{
                window.parent.gtag('js', new Date());
                window.parent.gtag('config', '{GA_ID}', {{
                    app_name: '{app_name}'
                }});
                window.parent.gtag('event', 'app_loaded', {{
                    app_name: '{app_name}'
                }});
            }}, 1000);
        }}
        </script>
        """,
        height=0,
    )


def track_event(event_name: str, category: str, label: str = "") -> None:
    components.html(
        f"""
        <script>
        const host = window.parent.location.hostname;
        const isLocal =
            host === "localhost" ||
            host === "127.0.0.1" ||
            host === "0.0.0.0";

        if (!isLocal && window.parent.gtag) {{
            window.parent.gtag('event', '{event_name}', {{
                event_category: '{category}',
                event_label: '{label}'
            }});
        }}
        </script>
        """,
        height=0,
    )