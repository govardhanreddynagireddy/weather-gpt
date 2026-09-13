from pathlib import Path
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from urllib.parse import urljoin
from urllib.request import Request,urlopen


# =====================================================
# PATHS
# =====================================================

BASE_DIR=Path(__file__).resolve().parent

DOCUMENTS_DIR=BASE_DIR/"documents"

BUILD_INDEX_FILE=BASE_DIR/"build_index.py"

DOCUMENTS_DIR.mkdir(exist_ok=True)


# =====================================================
# IMD SOURCE
# =====================================================

IMD_PAGE_URL=(
    "https://mausam.imd.gov.in/"
    "imd_latest/contents/"
    "all_india_forcast_bulletin.php"
)


# =====================================================
# DOWNLOAD SETTINGS
# =====================================================

USER_AGENT=(
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/153.0 Safari/537.36"
)


# =====================================================
# DOWNLOAD IMD PAGE
# =====================================================

def download_page():

    print("\nDownloading IMD bulletin page...")

    request=Request(
        IMD_PAGE_URL,
        headers={
            "User-Agent":USER_AGENT
        }
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        html=response.read().decode(
            "utf-8",
            errors="ignore"
        )

    print(
        f"IMD page downloaded: "
        f"{len(html)} characters"
    )

    return html


# =====================================================
# FIND ACTUAL AIWFB PDF
# =====================================================

def find_pdf_url(html):

    print("\nSearching for latest AIWFB PDF...")

    links=re.findall(
        r'href\s*=\s*["\']([^"\']+\.pdf[^"\']*)["\']',
        html,
        flags=re.IGNORECASE
    )

    if not links:

        raise RuntimeError(
            "Could not find any PDF link on the IMD page."
        )


    # Prefer AIWFB links

    preferred=[]

    for link in links:

        lower=link.lower()

        if "aiwfb" in lower:

            preferred.append(link)


    if preferred:

        pdf_link=preferred[0]

    else:

        # Fallback: weather/forecast PDF
        # but NEVER forecasting_sop.pdf

        weather_links=[]

        for link in links:

            lower=link.lower()

            if (
                ("weather" in lower or "forecast" in lower)
                and
                "sop" not in lower
            ):

                weather_links.append(link)


        if weather_links:

            pdf_link=weather_links[0]

        else:

            raise RuntimeError(
                "Could not identify the AIWFB PDF."
            )


    pdf_url=urljoin(
        IMD_PAGE_URL,
        pdf_link
    )


    print(
        f"AIWFB PDF URL:\n{pdf_url}"
    )

    return pdf_url


# =====================================================
# DOWNLOAD PDF
# =====================================================

def download_pdf(
    pdf_url,
    output_path
):

    print(
        "\nDownloading latest AIWFB PDF..."
    )

    request=Request(
        pdf_url,
        headers={
            "User-Agent":USER_AGENT
        }
    )

    with urlopen(
        request,
        timeout=60
    ) as response:

        data=response.read()


    if not data:

        raise RuntimeError(
            "Downloaded PDF is empty."
        )


    if not data.startswith(b"%PDF"):

        raise RuntimeError(
            "Downloaded file is not a valid PDF."
        )


    with open(
        output_path,
        "wb"
    ) as f:

        f.write(data)


    print(
        f"Downloaded: {output_path.name}"
    )

    print(
        f"Size: {len(data):,} bytes"
    )


# =====================================================
# FILE HASH
# =====================================================

def file_hash(path):

    sha256=hashlib.sha256()


    with open(
        path,
        "rb"
    ) as f:

        while True:

            block=f.read(
                1024*1024
            )

            if not block:

                break

            sha256.update(block)


    return sha256.hexdigest()


# =====================================================
# FIND EXISTING REAL AIWFB PDF
# =====================================================

def find_existing_pdf():

    pdf_files=list(
        DOCUMENTS_DIR.glob(
            "AIWFB_*.pdf"
        )
    )


    # Do not count temporary download

    pdf_files=[
        p
        for p in pdf_files
        if p.name!="AIWFB_latest_temp.pdf"
    ]


    if not pdf_files:

        return None


    pdf_files.sort(
        key=lambda p:p.stat().st_mtime,
        reverse=True
    )


    return pdf_files[0]


# =====================================================
# REMOVE OLD AIWFB FILES
# =====================================================

def remove_old_aiwfb_files(
    keep_file
):

    for pdf in DOCUMENTS_DIR.glob(
        "AIWFB_*.pdf"
    ):

        if pdf.name=="AIWFB_latest_temp.pdf":

            continue


        if pdf.resolve()!=keep_file.resolve():

            print(
                f"Removing old bulletin: "
                f"{pdf.name}"
            )

            pdf.unlink()


# =====================================================
# BUILD RAG INDEX
# =====================================================

def rebuild_index():

    print(
        "\n================================"
    )

    print(
        "Rebuilding RAG index..."
    )

    print(
        "================================"
    )


    if not BUILD_INDEX_FILE.exists():

        raise RuntimeError(
            f"build_index.py not found:\n"
            f"{BUILD_INDEX_FILE}"
        )


    result=subprocess.run(

        [
            sys.executable,
            str(BUILD_INDEX_FILE)
        ],

        cwd=str(BASE_DIR),

        text=True
    )


    if result.returncode!=0:

        raise RuntimeError(
            "RAG index building failed."
        )


    print(
        "\nRAG index updated successfully."
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print(
        "=========================================="
    )

    print(
        " WeatherGPT - IMD AIWFB Auto Updater"
    )

    print(
        "=========================================="
    )


    try:

        # 1. Download IMD page

        html=download_page()


        # 2. Find actual AIWFB PDF

        pdf_url=find_pdf_url(
            html
        )


        # 3. Download temporary PDF

        temp_path=(
            DOCUMENTS_DIR/
            "AIWFB_latest_temp.pdf"
        )


        download_pdf(
            pdf_url,
            temp_path
        )


        # 4. Calculate new hash

        new_hash=file_hash(
            temp_path
        )


        print(
            f"\nNew PDF SHA256:\n{new_hash}"
        )


        # 5. Find old bulletin

        existing=find_existing_pdf()


        if existing:

            old_hash=file_hash(
                existing
            )


            print(
                f"\nExisting PDF: "
                f"{existing.name}"
            )


            print(
                f"Existing SHA256:\n"
                f"{old_hash}"
            )


            # Same document

            if new_hash==old_hash:

                print(
                    "\nNo new IMD bulletin found."
                )


                temp_path.unlink()


                print(
                    "\nRAG index is already up to date."
                )


                return


        # 6. New bulletin

        print(
            "\nNEW IMD BULLETIN FOUND!"
        )


        # 7. Use today's date

        date_string=datetime.now().strftime(
            "%Y-%m-%d"
        )


        final_path=(
            DOCUMENTS_DIR/
            f"AIWFB_{date_string}.pdf"
        )


        # 8. Replace same-date file if necessary

        if final_path.exists():

            final_path.unlink()


        temp_path.rename(
            final_path
        )


        print(
            f"\nSaved new bulletin:\n"
            f"{final_path}"
        )


        # 9. Remove old bulletins

        remove_old_aiwfb_files(
            final_path
        )


        # 10. Rebuild RAG

        rebuild_index()


        print(
            "\n=========================================="
        )

        print(
            " IMD UPDATE COMPLETE"
        )

        print(
            "=========================================="
        )

        print(
            f"Latest bulletin: "
            f"{final_path.name}"
        )

        print(
            "FAISS index: UPDATED"
        )

        print(
            "RAG: UPDATED"
        )


    except Exception as error:

        print(
            "\n=========================================="
        )

        print(
            " IMD UPDATE FAILED"
        )

        print(
            "=========================================="
        )

        print(
            f"\nError:\n{error}"
        )

        sys.exit(1)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__=="__main__":

    main()