import { ModelConfiguration } from "@/components/model-configuration";
import { ThemeToggle } from "@/components/theme-toggle";

export default function Home() {
  return (
    <div className="relative flex min-h-screen items-center justify-center bg-background p-6 transition-colors md:p-8">
      <div className="fixed top-4 right-4 z-10 md:top-6 md:right-6">
        <ThemeToggle />
      </div>
      <ModelConfiguration />
    </div>
  );
}
