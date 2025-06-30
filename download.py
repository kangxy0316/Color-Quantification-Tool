#!/usr/bin/env python3
"""
Color Quantification Tool - Automatic Download Script
Automatically downloads the latest version from multiple mirrors
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import time

class ColorQuantificationDownloader:
    def __init__(self):
        self.filename = "ColorQuantificationTool.exe"
        self.expected_size = 1145549802  # 1.1GB in bytes
        self.expected_md5 = ""  # To be filled after upload
        self.expected_sha256 = ""  # To be filled after upload
        
        # Download mirrors (will be updated with actual links)
        self.download_urls = [
            {
                "name": "Google Drive",
                "url": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID&export=download",
                "priority": 1
            },
            {
                "name": "OneDrive", 
                "url": "https://onedrive.live.com/download?cid=YOUR_ONEDRIVE_CID&resid=YOUR_RESID",
                "priority": 2
            },
            {
                "name": "百度网盘 (Baidu)",
                "url": "https://pan.baidu.com/s/YOUR_SHARE_ID",
                "priority": 3,
                "note": "需要提取码: cqt1"
            },
            {
                "name": "SourceForge",
                "url": "https://sourceforge.net/projects/colorquantificationtool/files/latest/download",
                "priority": 4
            },
            {
                "name": "MediaFire",
                "url": "https://www.mediafire.com/file/YOUR_FILE_ID/ColorQuantificationTool.exe/file",
                "priority": 5
            }
        ]
        
        # Sort by priority
        self.download_urls.sort(key=lambda x: x["priority"])

    def print_banner(self):
        """Print welcome banner"""
        print("=" * 70)
        print("🎨 Color Quantification Tool - Automatic Downloader")
        print("=" * 70)
        print(f"📁 Target File: {self.filename}")
        print(f"📏 File Size: {self.expected_size / (1024**3):.1f} GB")
        print(f"🌐 Available Mirrors: {len(self.download_urls)}")
        print("=" * 70)

    def check_existing_file(self):
        """Check if file already exists and is valid"""
        if os.path.exists(self.filename):
            print(f"📁 Found existing file: {self.filename}")
            file_size = os.path.getsize(self.filename)
            print(f"📏 File size: {file_size / (1024**3):.1f} GB")
            
            if file_size == self.expected_size:
                print("✅ File size matches expected size")
                if self.verify_file_integrity():
                    print("✅ File integrity verified - download not needed!")
                    return True
                else:
                    print("❌ File integrity check failed - will re-download")
                    os.remove(self.filename)
            else:
                print("❌ File size mismatch - will re-download")
                os.remove(self.filename)
        
        return False

    def download_from_url(self, mirror):
        """Download file from a specific mirror"""
        print(f"\n🔄 Attempting download from: {mirror['name']}")
        
        if "note" in mirror:
            print(f"📝 Note: {mirror['note']}")
        
        try:
            # Handle special cases for different platforms
            if "drive.google.com" in mirror["url"]:
                return self.download_from_google_drive(mirror["url"])
            elif "pan.baidu.com" in mirror["url"]:
                print("⚠️  百度网盘需要手动下载，请访问链接并使用提取码: cqt1")
                print(f"🔗 链接: {mirror['url']}")
                return False
            else:
                return self.download_direct(mirror["url"], mirror["name"])
                
        except Exception as e:
            print(f"❌ Download failed from {mirror['name']}: {str(e)}")
            return False

    def download_direct(self, url, source_name):
        """Direct download with progress bar"""
        try:
            print(f"🌐 Connecting to {source_name}...")
            
            # Start download with stream=True for large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size from headers
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size == 0:
                print("⚠️  Warning: Could not determine file size")
                total_size = self.expected_size
            
            print(f"📏 Download size: {total_size / (1024**3):.1f} GB")
            
            # Download with progress bar
            with open(self.filename, 'wb') as f, tqdm(
                desc=f"📥 {source_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80
            ) as pbar:
                
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Download completed from {source_name}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Download error: {str(e)}")
            return False

    def download_from_google_drive(self, url):
        """Special handling for Google Drive downloads"""
        print("🔄 Handling Google Drive download...")
        
        try:
            # First request to get the download warning page
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Look for download confirmation token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            else:
                token = None
            
            if token:
                # Second request with confirmation token
                params = {'confirm': token}
                response = session.get(url, params=params, stream=True)
            
            return self.download_direct_stream(response, "Google Drive")
            
        except Exception as e:
            print(f"❌ Google Drive download failed: {str(e)}")
            return False

    def download_direct_stream(self, response, source_name):
        """Download from an existing response stream"""
        try:
            total_size = int(response.headers.get('content-length', self.expected_size))
            
            with open(self.filename, 'wb') as f, tqdm(
                desc=f"📥 {source_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ncols=80
            ) as pbar:
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"❌ Stream download error: {str(e)}")
            return False

    def verify_file_integrity(self):
        """Verify downloaded file integrity"""
        if not os.path.exists(self.filename):
            return False
        
        print("🔍 Verifying file integrity...")
        
        # Check file size
        file_size = os.path.getsize(self.filename)
        if file_size != self.expected_size:
            print(f"❌ Size mismatch: {file_size} != {self.expected_size}")
            return False
        
        # If we have checksums, verify them
        if self.expected_md5 or self.expected_sha256:
            print("🔐 Computing checksums...")
            
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            with open(self.filename, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            computed_md5 = md5_hash.hexdigest()
            computed_sha256 = sha256_hash.hexdigest()
            
            if self.expected_md5 and computed_md5 != self.expected_md5:
                print(f"❌ MD5 mismatch: {computed_md5} != {self.expected_md5}")
                return False
            
            if self.expected_sha256 and computed_sha256 != self.expected_sha256:
                print(f"❌ SHA256 mismatch: {computed_sha256} != {self.expected_sha256}")
                return False
            
            print("✅ Checksums verified")
        
        print("✅ File integrity verified")
        return True

    def run(self):
        """Main download process"""
        self.print_banner()
        
        # Check if file already exists and is valid
        if self.check_existing_file():
            return True
        
        print("\n🚀 Starting download process...")
        
        # Try each mirror in order of priority
        for mirror in self.download_urls:
            if self.download_from_url(mirror):
                # Verify the downloaded file
                if self.verify_file_integrity():
                    print(f"\n🎉 Successfully downloaded from {mirror['name']}!")
                    print(f"📁 File location: {os.path.abspath(self.filename)}")
                    print("✅ Ready to use!")
                    return True
                else:
                    print(f"❌ File verification failed for {mirror['name']}")
                    if os.path.exists(self.filename):
                        os.remove(self.filename)
                    continue
            
            print(f"❌ Failed to download from {mirror['name']}")
            time.sleep(2)  # Brief pause before trying next mirror
        
        print("\n❌ All download attempts failed!")
        print("📞 Please try manual download or contact support")
        return False

def main():
    """Main entry point"""
    try:
        downloader = ColorQuantificationDownloader()
        success = downloader.run()
        
        if success:
            print("\n🎯 Next steps:")
            print("1. Run ColorQuantificationTool.exe")
            print("2. Follow the user guide for detailed instructions")
            print("3. Enjoy your color analysis!")
        else:
            print("\n🆘 Need help?")
            print("- Check your internet connection")
            print("- Try running as administrator")
            print("- Visit our support page for manual download links")
        
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Download cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
