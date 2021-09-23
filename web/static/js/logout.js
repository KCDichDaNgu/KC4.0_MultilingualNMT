/*logout event*/
$("#id_topnav #log_out").click(function () {        
    $('#loader').show();
    $.ajax({
        type: "POST",
        url: "/log_out_exe",
        success: function (response) {
            $('#loader').hide();
            result = response["data"];
            window.location.href = result["msg"];
        },
        error: function (err) {
            $('#loader').hide();
            alert_error_ajax();
        }
    });
});
