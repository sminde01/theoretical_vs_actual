import os
import streamlit.components.v1 as components

GA_ID = "G-8DM8073S27"

def load_ga(app_name: str) -> None:
    components.html(
        f"""
        <script>
        (function() {{
            const w = window.parent;
            const host = w.location.hostname || "";
            const href = w.location.href || "";

            // Local/dev detection
            const isLocal =
                host === "localhost" ||
                host === "127.0.0.1" ||
                host === "0.0.0.0" ||
                host.endsWith(".local") ||
                host.startsWith("192.168.") ||
                host.startsWith("10.") ||
                host.startsWith("172.") ||
                href.includes("localhost");

            if (isLocal) {{
                console.log("GA skipped for local environment:", href);
                return;
            }}

            // Create/reuse persistent browser ID
            let clientId = w.localStorage.getItem("custom_ga_user_id");
            if (!clientId) {{
                clientId = "user_" + Math.random().toString(36).slice(2) + "_" + Date.now();
                w.localStorage.setItem("custom_ga_user_id", clientId);
            }}

            // Load GA script once
            const existingScript = w.document.querySelector(
                'script[src*="googletagmanager.com/gtag/js?id={GA_ID}"]'
            );

            if (!existingScript) {{
                const s = w.document.createElement("script");
                s.async = true;
                s.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
                w.document.head.appendChild(s);
            }}

            // Init gtag once
            w.dataLayer = w.dataLayer || [];
            function gtag() {{ w.dataLayer.push(arguments); }}
            w.gtag = w.gtag || gtag;

            if (!w.__ga_initialized__) {{
                w.__ga_initialized__ = true;

                setTimeout(() => {{
                    w.gtag('js', new Date());

                    // user_id is best used with your own stable identifier.
                    // Here we use a browser-persistent ID since there is no login.
                    w.gtag('config', '{GA_ID}', {{
                        user_id: clientId,
                        app_name: '{app_name}'
                    }});

                    // Fire app_loaded once per browser tab session for this app
                    const loadKey = "ga_app_loaded_" + "{app_name}";
                    const alreadyTracked = w.sessionStorage.getItem(loadKey);

                    if (!alreadyTracked) {{
                        w.gtag('event', 'app_loaded', {{
                            app_name: '{app_name}',
                            user_id: clientId
                        }});
                        w.sessionStorage.setItem(loadKey, "true");
                    }}
                }}, 800);
            }}
        }})();
        </script>
        """,
        height=0,
    )


def track_event(event_name: str, category: str, label: str = "") -> None:
    # basic JS escaping for single quotes
    event_name = event_name.replace("'", "\\'")
    category = category.replace("'", "\\'")
    label = label.replace("'", "\\'")

    components.html(
        f"""
        <script>
        (function() {{
            const w = window.parent;
            const host = w.location.hostname || "";
            const href = w.location.href || "";

            const isLocal =
                host === "localhost" ||
                host === "127.0.0.1" ||
                host === "0.0.0.0" ||
                host.endsWith(".local") ||
                host.startsWith("192.168.") ||
                host.startsWith("10.") ||
                host.startsWith("172.") ||
                href.includes("localhost");

            if (isLocal || !w.gtag) return;

            let clientId = w.localStorage.getItem("custom_ga_user_id");
            if (!clientId) {{
                clientId = "user_" + Math.random().toString(36).slice(2) + "_" + Date.now();
                w.localStorage.setItem("custom_ga_user_id", clientId);
            }}

            w.gtag('event', '{event_name}', {{
                event_category: '{category}',
                event_label: '{label}',
                user_id: clientId
            }});
        }})();
        </script>
        """,
        height=0,
    )
# def load_ga(app_name: str) -> None:
#     components.html(
#         f"""
#         <script>
#         const host = window.parent.location.hostname;
#         const isLocal =
#             host === "localhost" ||
#             host === "127.0.0.1" ||
#             host === "0.0.0.0";

#         if (!isLocal) {{
#             const existingScript = window.parent.document.querySelector(
#                 'script[src*="googletagmanager.com/gtag/js?id={GA_ID}"]'
#             );

#             if (!existingScript) {{
#                 const s = window.parent.document.createElement("script");
#                 s.async = true;
#                 s.src = "https://www.googletagmanager.com/gtag/js?id={GA_ID}";
#                 window.parent.document.head.appendChild(s);
#             }}

#             window.parent.dataLayer = window.parent.dataLayer || [];
#             function gtag() {{ window.parent.dataLayer.push(arguments); }}
#             window.parent.gtag = window.parent.gtag || gtag;

#             setTimeout(() => {{
#                 window.parent.gtag('js', new Date());
#                 window.parent.gtag('config', '{GA_ID}', {{
#                     app_name: '{app_name}'
#                 }});
#                 window.parent.gtag('event', 'app_loaded', {{
#                     app_name: '{app_name}'
#                 }});
#             }}, 1000);
#         }}
#         </script>
#         """,
#         height=0,
#     )


# def track_event(event_name: str, category: str, label: str = "") -> None:
#     components.html(
#         f"""
#         <script>
#         const host = window.parent.location.hostname;
#         const isLocal =
#             host === "localhost" ||
#             host === "127.0.0.1" ||
#             host === "0.0.0.0";

#         if (!isLocal && window.parent.gtag) {{
#             window.parent.gtag('event', '{event_name}', {{
#                 event_category: '{category}',
#                 event_label: '{label}'
#             }});
#         }}
#         </script>
#         """,
#         height=0,
#     )