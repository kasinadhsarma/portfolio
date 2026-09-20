export function isAuthorizedWrite(request: Request): boolean {
  const secret = process.env.API_WRITE_SECRET
  if (!secret) return false

  const header = request.headers.get('authorization')
  return header === `Bearer ${secret}`
}
