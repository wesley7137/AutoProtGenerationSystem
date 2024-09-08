import React, { useState, useEffect } from "react";
import { Snackbar } from "@material-ui/core";
import { Alert } from "@material-ui/lab";

function Notifications() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [severity, setSeverity] = useState("info");

  useEffect(() => {
    const eventSource = new EventSource("/api/events");

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessage(data.message);
      setSeverity(data.severity);
      setOpen(true);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const handleClose = (event, reason) => {
    if (reason === "clickaway") {
      return;
    }
    setOpen(false);
  };

  return (
    <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
      <Alert onClose={handleClose} severity={severity}>
        {message}
      </Alert>
    </Snackbar>
  );
}

export default Notifications;
