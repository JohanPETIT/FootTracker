mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"alexis.kim--tan@edu.esieee.fr\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
maxUploadSize = 7000\n\
\n\
[client]\n\
showSidebarNavigation = false\n\
\n\
[theme]\n\
base = \"dark\"\n\
" > ~/.streamlit/config.toml
