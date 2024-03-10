<?php
// Simulate getting LaTeX content from a database or file
$latexContent = "Your LaTeX content goes here.";

// Set appropriate headers for downloading a text file
header('Content-Type: text/plain');
header('Content-Disposition: attachment; filename=latex_content.txt');

// Output the LaTeX content
echo $latexContent;
