import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import crypto from 'crypto'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Function to create a secret key (in a real application, this should be in an environment variable)
const SECRET_KEY = 'your-secret-key'

// Function to encrypt URL
export function encryptUrl(url: string): string {
  try {
    // Create hash of URL for shorter encoded strings
    const hash = crypto.createHash('sha256')
    hash.update(url + SECRET_KEY)
    return hash.digest('base64url')
  } catch (error) {
    console.error('Error encrypting URL:', error)
    return encodeURIComponent(url) // Fallback to simple encoding
  }
}

// URL storage map
const urlMap = new Map<string, string>()

// Function to store URL and get encoded version
export function storeAndEncodeUrl(url: string): string {
  const encoded = encryptUrl(url)
  urlMap.set(encoded, url)
  return encoded
}

// Function to decode stored URL
export function decodeStoredUrl(encoded: string): string | null {
  return urlMap.get(encoded) || null
}

// Function to safely open URL
export function safeOpenUrl(encodedUrl: string, target: string = '_blank'): void {
  const url = decodeStoredUrl(encodedUrl)
  if (url) {
    window.open(url, target, 'noopener noreferrer')
  } else {
    console.error('Invalid or expired URL')
  }
}
