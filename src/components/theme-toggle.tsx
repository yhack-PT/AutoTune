"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

type Theme = "light" | "dark";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={() =>
        setTheme((currentTheme) =>
          currentTheme === "dark" ? "light" : "dark"
        )
      }
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      className="shrink-0"
    >
      {theme === "dark" ? <Sun /> : <Moon />}
      {theme === "dark" ? "Light Mode" : "Dark Mode"}
    </Button>
  );
}
