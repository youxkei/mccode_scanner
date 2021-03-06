(async () => {
  const { scan } = await import("../pkg/index.js").catch(console.error);

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const result = document.getElementById("result");
  const rescan = document.getElementById("rescan");

  const context = canvas.getContext("2d");

  const scanWidth = 360;
  const constraint = {
    video: {
      facingMode: { exact: "environment" },
      width: { exact: (scanWidth * 8) / 3 },
      height: { exact: scanWidth * 2 },
    },
    audio: false,
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraint);

    video.srcObject = stream;
    video.play();

    function main() {
      if (video.readyState >= 2) {
        const width = video.videoWidth;
        const height = video.videoHeight;

        canvas.width = width;
        canvas.height = height;

        context.drawImage(video, 0, 0);
        context.strokeRect(
          (width - scanWidth) / 2,
          (height - scanWidth) / 2,
          scanWidth,
          scanWidth
        );

        try {
          const scanResult = scan(
            scanWidth,
            scanWidth,
            context.getImageData(
              (width - scanWidth) / 2,
              (height - scanWidth) / 2,
              scanWidth,
              scanWidth
            ).data
          );

          if (scanResult) {
            result.textContent = scanResult;
          } else {
            window.requestAnimationFrame(main);
          }
        } catch (e) {
          console.error(e);
          window.requestAnimationFrame(main);
        }
      } else {
        window.requestAnimationFrame(main);
      }
    }

    rescan.addEventListener("click", () => {
      result.textContent = "";
      window.requestAnimationFrame(main);
    });

    window.requestAnimationFrame(main);
  } catch (e) {
    console.error(e);
  }
})();
