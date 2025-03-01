import Image from 'next/image'

export const GitHubViews = () => {
  return (
    <div className="fixed bottom-4 right-4 z-50">
      <a
        href="https://github.com/kasinadhsarma/portfolio"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 rounded-lg bg-background/95 p-2 text-sm text-muted-foreground shadow-lg backdrop-blur"
      >
        <Image
          alt="GitHub Views"
          width={150}
          height={20}
          src="https://komarev.com/ghpvc/?username=kasinadhsarma&label=Portfolio+Views"
        />
      </a>
    </div>
  )
}
