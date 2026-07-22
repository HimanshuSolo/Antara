"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/results", label: "Results" },
  { href: "/demo", label: "Demo" },
  { href: "/live", label: "Live" },
];

export default function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="container container--wide site-header__inner">
        <Link href="/" className="site-header__mark">
          Antara
        </Link>
        <nav className="site-header__nav">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={pathname === link.href ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
