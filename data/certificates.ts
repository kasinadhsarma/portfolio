export interface Certificate {
  title: string;
  issuer: string;
  date: string;
  image: string;
  url: string;
  category: "featured" | "cloud" | "work" | "practical";
}

export const certificates: Certificate[] = [
  {
    title: "Intel Unnati Program Certificate",
    issuer: "Intel",
    date: "2023",
    image: "/img/1691495618292.jpeg",
    url: "https://github.com/Exploit0xfffff/Intelrepo",
    category: "practical"
  },
  {
    title: "Embedded Systems Training",
    issuer: "Training Certificate",
    date: "2024",
    image: "/img/1710515420773_page-0001.jpg",
    url: "https://drive.google.com/file/d/1-uYXufD1HmSBbj_cIH8lPbkc21jD-r7a/view?usp=sharing",
    category: "practical"
  },
  {
    title: "Industrial Training",
    issuer: "Training Certificate",
    date: "2024",
    image: "/img/1710515538186_page-0001.jpg",
    url: "https://drive.google.com/file/d/1DJMxonp4K-Xqh5mEGPcXPQ2HDRl_4Bv7/view?usp=sharing",
    category: "practical"
  },
  {
    title: "Google Cloud Computing Foundations",
    issuer: "Google Cloud",
    date: "2023",
    image: "/img/google-cloud-computing-foundations-certificate.png",
    url: "https://www.credly.com/badges/41109701-f95c-4359-abbd-56238e06f956/public_url",
    category: "featured"
  },
  {
    title: "Google Cybersecurity Certificate",
    issuer: "Google",
    date: "2023",
    image: "/img/google-cybersecurity-certificate.png",
    url: "https://www.credly.com/badges/ccd57157-f61f-470f-84bb-c64830652dc4/",
    category: "featured"
  },
  {
    title: "JPMorgan Chase Software Engineering",
    issuer: "JPMorgan Chase & Co.",
    date: "2024",
    image: "/img/jpmorgen.jpg",
    url: "https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/J.P.%20Morgan/gWbW5qHAChqQBGWpA_JPMorgan%20Chase%20&%20Co._WtuAmSzX4xtRAH2dM_1717079478246_completion_certificate.pdf",
    category: "featured"
  },
  {
    title: "AIG Cybersecurity Virtual Experience",
    issuer: "AIG",
    date: "2024",
    image: "/img/AIG-1.png",
    url: "https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/AIG/2ZFnEGEDKTQMtEv9C_AIG_WtuAmSzX4xtRAH2dM_1714632980200_completion_certificate.pdf",
    category: "featured"
  },
  {
    title: "Mastercard Virtual Experience",
    issuer: "Mastercard",
    date: "2024",
    image: "/img/mastercard-1.png",
    url: "https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/mastercard/vcKAB5yYAgvemepGQ_Mastercard_WtuAmSzX4xtRAH2dM_1712296014916_completion_certificate.pdf",
    category: "featured"
  },
  {
    title: "Accenture Virtual Experience",
    issuer: "Accenture UK",
    date: "2024",
    image: "/img/accenture-1.png",
    url: "https://forage-uploads-prod.s3.amazonaws.com/completion-certificates/Accenture%20UK/EzKFRQ2oEA87PPjsL_Accenture%20UK_WtuAmSzX4xtRAH2dM_1714062765236_completion_certificate.pdf",
    category: "featured"
  }
];

export const cloudBadges: Certificate[] = [
  {
    title: "Cloud Development",
    issuer: "Google Cloud",
    date: "2023",
    image: "/img/CD.png",
    url: "https://www.cloudskillsboost.google/public_profiles/1751e99e-eedf-4653-a271-075574b8be67/badges/6669681",
    category: "cloud"
  },
  {
    title: "Google Workspace",
    issuer: "Google Cloud",
    date: "2023",
    image: "/img/googleworkspace.png",
    url: "https://www.cloudskillsboost.google/public_profiles/1751e99e-eedf-4653-a271-075574b8be67/badges/6703272",
    category: "cloud"
  },
  {
    title: "Cloud Security",
    issuer: "Google Cloud",
    date: "2023",
    image: "/img/mitigate.png",
    url: "https://www.cloudskillsboost.google/public_profiles/1751e99e-eedf-4653-a271-075574b8be67/badges/6707759",
    category: "cloud"
  },
  {
    title: "App Development",
    issuer: "Google Cloud",
    date: "2023",
    image: "/img/appsheet.png",
    url: "https://www.cloudskillsboost.google/public_profiles/1751e99e-eedf-4653-a271-075574b8be67/badges/6720261",
    category: "cloud"
  }
];
