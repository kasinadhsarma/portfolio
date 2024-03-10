function downloadLatex() {
    // Create a download link
    const downloadLink = document.createElement('a');
    downloadLink.href = 'download.php';
    downloadLink.download = 'latex_content.txt';

    // Append the link to the document and trigger the download
    document.body.appendChild(downloadLink);
    downloadLink.click();

    // Remove the link from the document
    document.body.removeChild(downloadLink);
}
