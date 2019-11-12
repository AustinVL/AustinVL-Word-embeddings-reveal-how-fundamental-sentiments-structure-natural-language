function(req) {
  fluidPage(
    # set up -----
    title = "EPA Ratings",
    tags$head(tags$link(href = "browser_tab_icon.png", rel = "icon")),
    
    includeCSS("www/style.css"),
    
    useShinyjs(),
    
    h1("EPA Ratings"),
    hr(),
    
    # fetch IP -----
    div(
      style = "display: none;",
      textInput(
        inputId = "remote_addr",
        label = "remote_addr",
        value = if (!is.null(req[["HTTP_X_FORWARDED_FOR"]])) {
          req[["HTTP_X_FORWARDED_FOR"]]
        } else {
          req[["REMOTE_ADDR"]]
        }
      )
    ),
    
    # info and settings -----
    fluidRow(
      actionLink(inputId = "info", label = NULL, icon = icon("info-circle"))
    ),
    br(),
    
    # login panel -----
    fluidRow(
      column(width = 12, align = "center", uiOutput(outputId = "login_panel"))
    ),
    br(), br(),
    
    # message panel -----
    fluidRow(
      column(width = 12, align = "center", htmlOutput(outputId = "message"))
    ),
    
    # ratings panel -----
    uiOutput(outputId = "ratings_panel")
  )
}
