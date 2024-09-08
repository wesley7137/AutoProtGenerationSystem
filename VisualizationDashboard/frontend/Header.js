import React from "react";
import { AppBar, Toolbar, Typography, Button } from "@material-ui/core";
import { Link } from "react-router-dom";

function Header() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" style={{ flexGrow: 1 }}>
          Protein Engineering Dashboard
        </Typography>
        <Button color="inherit" component={Link} to="/">
          Dashboard
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
