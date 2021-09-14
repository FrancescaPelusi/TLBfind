#awk -v time=${TIME} -f select.awk
BEGIN{k = 0}
{
    if(NF!= 0 && $1 == time){
	print($7, $8, $9, $10);
    }
}END{}