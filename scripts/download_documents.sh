l#!/bin/bash
# download_documents.sh: Baixa o arquivo ZIP do Dropbox e extrai seu conteúdo.

ZIP_URL="https://www.dropbox.com/scl/fo/567bfz020h4j6m88t6cvi/AKOl4CYf5hC_P4IHoPSeMDk?rlkey=qykon4ze1j37e5llx5n4xbozp&st=msg3vf35&dl=1"
ZIP_FILE="documents.zip"
DEST_DIR="./documents"

echo "[SCRIPT LOG] Downloading $ZIP_FILE from Dropbox..."
wget "$ZIP_URL" -O "$ZIP_FILE" || {
  echo "[SCRIPT LOG] Error on downloading."
  exit 1
}

echo "[SCRIPT LOG] Creating directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "[SCRIPT LOG] Extracting content from $ZIP_FILE to $DEST_DIR..."
unzip "$ZIP_FILE" -d "$DEST_DIR"
EXIT_CODE=$?

# Se o unzip retornar código maior que 2, consideramos um erro crítico e encerramos
if [ $EXIT_CODE -gt 2 ]; then
    echo "[SCRIPT LOG] Error on extracting file (exit code $EXIT_CODE)."
    exit $EXIT_CODE
fi

# Se o unzip retornar 2, consideramos apenas um aviso (por exemplo, caminhos absolutos)
if [ $EXIT_CODE -eq 2 ]; then
    echo "[SCRIPT LOG] Warning: unzip returned code 2."
fi

echo "[SCRIPT LOG] Download and extraction completed!"

echo "[SCRIPT LOG] Deleting zip file..."
rm "$ZIP_FILE"

echo "[SCRIPT LOG] End of <download_documents.sh> script"
